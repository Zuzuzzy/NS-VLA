"""Bi-granular advantage-weighted update: H-GRPO (high) + AWR (low) (Eq. 9 / Alg.1).

High level (H-GRPO): group-relative, trajectory-level weight
    w^h_i ∝ exp((R^h_i - mu_G) / (sigma_G + eps))          [group of G rollouts]
    L^h(phi) = - E[ w^h_i * sum_t log pi^h_phi(u_t | H_t) ] + beta_kl * KL(pi^h || pi^{h,*})
Low level (AWR): chunk-level weight
    w^l_t ∝ exp(A^l_t / beta_l),   L^l(theta) = E[ w^l_t * || pi^l_theta(e_t) - A_t ||_1 ]
Joint: L = L^h(phi) + lambda_l * L^l(theta).

Only the small trainable set Theta = {phi (pointer MLP), theta (the solver's
adapter + action head)} is updated; the solver's own backbone stays frozen. The KL anchor
to the frozen BC reference bounds high-level drift - **Proposition 3**.

The advantage / weight / KL math here is pure tensor code; the rollout coupling lives
in ``train_rl.py``.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def group_normalized_advantages(
    returns: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """A_i = (R_i - mu_G) / (sigma_G + eps) over a group of G rollouts (Eq. 9, w^h)."""
    r = returns.float()
    mu = r.mean()
    sigma = r.std(unbiased=False)
    return (r - mu) / (sigma + eps)


def high_level_weights(returns: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """w^h_i ∝ exp((R^h_i - mu_G)/(sigma_G + eps)), normalized to sum 1 over the group."""
    adv = group_normalized_advantages(returns, eps)
    w = torch.softmax(adv, dim=0)
    return w


def awr_chunk_weights(advantages: torch.Tensor, beta_l: float = 1.0) -> torch.Tensor:
    """w^l_t ∝ exp(A^l_t / beta_l), normalized over the rollout (AWR low-level, Eq. 9)."""
    return torch.softmax(advantages.float() / max(beta_l, 1e-6), dim=0)


def kl_categorical(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """KL(p || q) between two categorical policies given logits. >= 0, and 0 iff p == q.

    Masked-support safe. The high-level policy of Eq. 5 masks the inadmissible slot
    with ``-inf`` at the end of a plan (only "stay" is admissible), which makes
    ``p_i = 0`` and ``logp_i - logq_i = -inf - (-inf) = nan`` there; the convention
    ``0 * log 0 = 0`` must be applied explicitly or the whole objective turns into
    NaN and the optimizer writes NaN into phi.
    """
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    p = logp.exp()
    # sanitize the log-ratio BEFORE multiplying: ``p * nan`` would be nan even where
    # p == 0, and torch.where applied afterwards still leaks a nan *gradient* into p.
    diff = logp - logq
    diff = torch.where(torch.isfinite(diff), diff, torch.zeros_like(diff))
    return torch.where(p > 0, p * diff, torch.zeros_like(p)).sum(dim=-1)


def high_level_loss(
    slot_logprobs: torch.Tensor,   # (G, T) log pi^h(u_t | H_t) along each rollout
    returns: torch.Tensor,          # (G,) high-level returns R^h
    pi_logits: torch.Tensor,        # (N, V) current high-level logits (for KL)
    ref_logits: torch.Tensor,       # (N, V) frozen BC-anchor logits pi^{h,*}
    beta_kl: float = 0.05,
    eps: float = 1e-6,
) -> torch.Tensor:
    """L^h(phi) = - E[w^h_i sum_t log pi^h] + beta_kl * KL(pi^h || pi^{h,*}) (Eq. 9)."""
    w = high_level_weights(returns, eps).detach()          # (G,) stop-grad on the weight
    pg = -(w * slot_logprobs.sum(dim=1)).sum()
    kl = kl_categorical(pi_logits, ref_logits).mean()
    return pg + beta_kl * kl


def low_level_loss(
    pred_chunks: torch.Tensor,      # (T, H, action_dim) pi^l_theta(e_t)
    target_chunks: torch.Tensor,    # (T, H, action_dim) executed chunks A_t
    advantages: torch.Tensor,       # (T,) chunk-level A^l_t
    beta_l: float = 1.0,
) -> torch.Tensor:
    """L^l(theta) = E[ w^l_t || pi^l_theta(e_t) - A_t ||_1 ] (AWR, Eq. 9)."""
    w = awr_chunk_weights(advantages, beta_l).detach()     # (T,)
    l1 = (pred_chunks - target_chunks).abs().flatten(1).sum(dim=1)   # (T,)
    return (w * l1).sum()


def joint_loss(l_high: torch.Tensor, l_low: torch.Tensor, lambda_l: float = 1.0) -> torch.Tensor:
    """L_joint(Theta) = L^h(phi) + lambda_l * L^l(theta) (Eq. 9)."""
    return l_high + lambda_l * l_low


class KLController:
    """Optional adaptive beta_kl keeping KL inside the target band [1e-3, 5e-2]."""

    def __init__(self, beta: float = 0.05, band: tuple[float, float] = (1e-3, 5e-2), rate: float = 1.5):
        self.beta = beta
        self.lo, self.hi = band
        self.rate = rate

    def update(self, kl: float) -> float:
        if kl > self.hi:
            self.beta *= self.rate
        elif kl < self.lo:
            self.beta /= self.rate
        return self.beta


# --------------------------------------------------------------------------- #
# chunk-level advantage (Eq. 9, low level: "A^l_t formed from r^l_t relative to a
# running baseline")
# --------------------------------------------------------------------------- #
class RunningBaseline:
    """EMA baseline b <- rho*b + (1-rho)*mean(r) giving A^l_t = r^l_t - b."""

    def __init__(self, momentum: float = 0.9, value: float = 0.0, initialized: bool = False):
        self.momentum = momentum
        self.value = value
        self.initialized = initialized

    def update(self, rewards: torch.Tensor) -> float:
        m = float(rewards.float().mean())
        if not self.initialized:
            self.value = m
            self.initialized = True
        else:
            self.value = self.momentum * self.value + (1.0 - self.momentum) * m
        return self.value

    def advantages(self, rewards: torch.Tensor, clip: float | None = None) -> torch.Tensor:
        a = rewards.float() - self.value
        if clip is not None:
            a = a.clamp(-clip, clip)
        return a

    def state_dict(self) -> dict:
        return {"momentum": self.momentum, "value": self.value, "initialized": self.initialized}

    def load_state_dict(self, d: dict) -> None:
        self.momentum = float(d.get("momentum", self.momentum))
        self.value = float(d.get("value", 0.0))
        self.initialized = bool(d.get("initialized", False))


# --------------------------------------------------------------------------- #
# high-level (phi) optimizer plumbing — H-GRPO with a frozen BC anchor
# --------------------------------------------------------------------------- #
class HighLevelUpdater:
    """Own the pointer classifier ``g_phi``, its frozen BC copy, optimizer and schedule.

    Mechanism (red line): ONLY ``phi`` moves; the frozen reference ``pi^{h,*}`` is a
    deep copy of the Stage-I BC pointer taken once at construction and never updated,
    and the {stay, +1} admissible mask inside ``PrimitiveClassifier`` is untouched —
    the KL and the policy gradient are both evaluated through that same masked
    2-way categorical, so monotonicity survives every update by construction.
    """

    def __init__(
        self,
        clf,
        lr: float = 3e-5,
        warmup_steps: int = 100,
        schedule: str = "cosine",
        total_steps: int = 50,
        beta_kl: float = 0.05,
        kl_band: tuple[float, float] = (1e-3, 5e-2),
        use_kl_controller: bool = True,
        grad_clip: float = 1.0,
        device: str = "cpu",
    ):
        import copy

        self.clf = clf.to(device)
        self.clf.train()
        self.ref = copy.deepcopy(clf).to(device).eval()
        for p in self.ref.parameters():
            p.requires_grad_(False)
        self.device = device
        self.base_lr = lr
        self.warmup_steps = max(0, warmup_steps)
        self.schedule = schedule
        self.total_steps = max(1, total_steps)
        self.grad_clip = grad_clip
        self.opt = torch.optim.AdamW(self.clf.parameters(), lr=lr)
        self.kl_ctl = KLController(beta_kl, kl_band) if use_kl_controller else None
        self.beta_kl = beta_kl
        self.step_count = 0

    # ---------------------------------------------------------------- #
    def lr_at(self, step: int) -> float:
        import math

        if self.warmup_steps and step < self.warmup_steps:
            return self.base_lr * (step + 1) / self.warmup_steps
        if self.schedule != "cosine":
            return self.base_lr
        prog = min(1.0, max(0.0, (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)))
        return 0.5 * self.base_lr * (1.0 + math.cos(math.pi * prog))

    # Finite stand-in for the -inf mask of Eq. 5, used as a belt-and-braces second line
    # of defence. The forward matches -inf far beyond float32 precision (softmax mass
    # ~e^-1e9 == 0) while keeping the BACKWARD finite. The FIRST line of defence is the
    # analytic degenerate-row handling below — see ``_masked_logp_kl``.
    MASK_LOGIT = -1e9

    def _cand_logits(self, model, psi, plan_op_ids, m_prev, plan_len):
        """Masked {stay, +1} candidate logits of Eq. 5 -> (logits (N,2), cand, degenerate).

        ``degenerate`` marks the rows where the plan pointer already sits on the last
        slot, so the admissible set K(m_{t-1}) collapses to a SINGLE element {stay}.
        """
        vocab_logits = model.logits(psi)
        cand, dup = model._candidates(m_prev, plan_len)
        cand_ops = torch.gather(plan_op_ids, 1, cand)
        cand_logits = torch.gather(vocab_logits, 1, cand_ops)
        bias = torch.tensor([0.0, model.advance_bias], dtype=cand_logits.dtype,
                            device=cand_logits.device)
        cand_logits = cand_logits + bias
        mask = torch.stack([torch.zeros_like(dup), dup], dim=1)
        return cand_logits.masked_fill(mask, self.MASK_LOGIT), cand, dup

    @staticmethod
    def _degenerate_zero(x: torch.Tensor, degenerate: torch.Tensor) -> torch.Tensor:
        """Force the analytic value 0 on single-candidate decision steps.

        When ``|K(m_{t-1})| == 1`` the high-level policy of Eq. 5 is a **Dirac**: the
        only admissible slot is taken with probability exactly 1, so

            log pi^h(u_t | H_t) = log 1 = 0        and      KL(pi^h || pi^{h,*}) = 0

        identically, for ANY phi. Those steps therefore carry no gradient signal for
        the high-level factor and must contribute exactly nothing to L^h. Computing
        them numerically instead - softmax over ``[x, -inf]`` - yields ``0 * log 0``
        inside the KL and a ``-inf`` log-prob if the "stay" is mislabelled, and the
        resulting NaN propagates into every parameter of phi in one optimizer step.
        Zeroing here is not a numerical guard, it is the analytic value.
        """
        return torch.where(degenerate, torch.zeros_like(x), x)

    def update(
        self,
        psi: torch.Tensor,          # (N, d_in) visited pooled features
        plan_op_ids: torch.Tensor,  # (N, M)
        m_prev: torch.Tensor,       # (N,)
        plan_len: torch.Tensor,     # (N,)
        taken_slot: torch.Tensor,   # (N,) slot actually executed
        rollout_index: torch.Tensor,  # (N,) which of the G rollouts each step belongs to
        returns: torch.Tensor,      # (G,) high-level returns R^h
    ) -> dict:
        """One H-GRPO gradient step on phi. Returns a metrics dict (all floats)."""
        psi = psi.to(self.device)
        plan_op_ids = plan_op_ids.to(self.device)
        m_prev = m_prev.to(self.device)
        plan_len = plan_len.to(self.device)
        taken_slot = taken_slot.to(self.device)
        rollout_index = rollout_index.to(self.device)
        returns = returns.to(self.device).float()

        G = int(returns.shape[0])
        w = high_level_weights(returns).detach()                       # (G,)
        adv = group_normalized_advantages(returns).detach()

        cand_logits, cand, degen = self._cand_logits(
            self.clf, psi, plan_op_ids, m_prev, plan_len
        )
        logp_all = F.log_softmax(cand_logits, dim=1)                   # (N, 2)
        # "advanced" must be read off the pointer MOVING, not off ``taken == m_next``:
        # at the last slot m_next == m_prev, so the latter test would mislabel a *stay*
        # as an advance and gather the masked candidate, giving L^h = +inf.
        is_advance = (taken_slot != m_prev).long()
        logp = torch.gather(logp_all, 1, is_advance[:, None]).squeeze(1)   # (N,)
        logp = self._degenerate_zero(logp, degen)      # Dirac step => log pi == 0

        # sum_t log pi per rollout, then the group weight
        per_rollout = torch.zeros(G, dtype=logp.dtype, device=self.device)
        per_rollout = per_rollout.index_add(0, rollout_index, logp)
        pg = -(w * per_rollout).sum()

        with torch.no_grad():
            ref_logits, _, _ = self._cand_logits(self.ref, psi, plan_op_ids, m_prev, plan_len)
        kl_vec = self._degenerate_zero(
            kl_categorical(cand_logits, ref_logits), degen               # Dirac => KL == 0
        )
        kl = kl_vec.mean()
        # The *reported / controlled* KL must be the mean over LIVE decision steps only.
        # Averaging the analytic zeros of degenerate rows in dilutes the statistic by the
        # live fraction, and on an all-degenerate iteration (a plan of length 1, where the
        # pointer structurally cannot move) it yields exactly 0 -- which the controller
        # below would read as "policy is glued to the anchor, relax it". Several LIBERO
        # tasks have M=1 plans, so that would fire regularly and ratchet beta_kl down
        # geometrically until the anchor no longer binds. Same ruling as L^h and the
        # gradient: degenerate rows are DROPPED, not counted as zeros.
        n_live = int((~degen).sum())
        kl_live = (kl_vec.sum() / max(n_live, 1)) if n_live else kl_vec.sum() * 0.0

        loss = pg + self.beta_kl * kl
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.clf.parameters(), self.grad_clip)
        lr = self.lr_at(self.step_count)
        for g in self.opt.param_groups:
            g["lr"] = lr
        self.opt.step()
        self.step_count += 1

        kl_val = float(kl_live.detach())
        # An all-degenerate iteration carries NO evidence about drift -> do not adapt.
        if self.kl_ctl is not None and n_live > 0:
            self.beta_kl = self.kl_ctl.update(kl_val)

        return {
            "loss_high": float(loss.detach()),
            "pg_high": float(pg.detach()),
            "kl": kl_val,
            "kl_all_rows": float(kl.detach()),
            "n_live_decisions": n_live,
            "beta_kl": float(self.beta_kl),
            "lr_high": float(lr),
            "grad_norm_high": float(gnorm),
            "adv_high_mean": float(adv.mean()),
            "adv_high_std": float(adv.std(unbiased=False)),
            "w_high_max": float(w.max()),
            "n_decisions": int(psi.shape[0]),
            # decision steps whose admissible set had collapsed to {stay}: analytic
            # zeros, i.e. steps that carry no high-level learning signal at all
            "n_degenerate_steps": int(degen.sum()),
        }

    # ---------------------------------------------------------------- #
    def kl_to_ref(self, psi, plan_op_ids, m_prev, plan_len) -> float:
        """Monitoring-only KL(pi^h || pi^{h,*}) on a batch of visited states."""
        with torch.no_grad():
            args = (psi.to(self.device), plan_op_ids.to(self.device),
                    m_prev.to(self.device), plan_len.to(self.device))
            cur, _, degen = self._cand_logits(self.clf, *args)
            ref, _, _ = self._cand_logits(self.ref, *args)
            return float(self._degenerate_zero(kl_categorical(cur, ref), degen).mean())

    def state_dict(self) -> dict:
        return {
            "clf": self.clf.state_dict(),
            "opt": self.opt.state_dict(),
            "step_count": self.step_count,
            "beta_kl": self.beta_kl,
        }

    def load_state_dict(self, d: dict) -> None:
        self.clf.load_state_dict(d["clf"])
        self.opt.load_state_dict(d["opt"])
        self.step_count = int(d.get("step_count", 0))
        self.beta_kl = float(d.get("beta_kl", self.beta_kl))
