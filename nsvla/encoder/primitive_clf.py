"""Plan-constrained primitive inference with a monotone pointer (NS-VLA §4.1, Eq. 5).

The classifier ``g_phi`` maps the pooled VLM context feature ``psi_bar`` to logits
over the primitive vocabulary. At each decision step the pointer may only *stay*
or *advance by one slot* — the admissible set ``K(m_{t-1}) = {m_{t-1},
min(m_{t-1}+1, M-1)}``. This mask makes **Proposition 1** (pointer monotone,
episode partitioned into <= M ordered segments) hold *by construction*,
independent of the learned weights.

The admissible set is the mechanism: widening it would void Proposition 1, so the
mask must not be relaxed. Architecture: LayerNorm -> MLP with hidden = 2*d_in, GELU.

``advance_bias`` is a *static* scalar added to the advance candidate's logit inside
the SAME argmax of Eq. 5 - an offline-fixed calibration constant that is part of
``g_phi``. It reads nothing outside ``psi_bar``, adds no gate and no fallback, and
cannot break monotonicity because the admissible set is still {stay, +1}. Negative
values make the pointer more conservative; 0.0 is the uncalibrated decision rule.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrimitiveClassifier(nn.Module):
    """LayerNorm -> Linear -> GELU -> Linear over the primitive vocabulary (Eq. 5, g_phi)."""

    def __init__(
        self,
        d_in: int,
        vocab_size: int,
        hidden: int | None = None,
        n_layers: int = 2,
        advance_bias: float = 0.0,
    ):
        super().__init__()
        if hidden is None:
            hidden = 2 * d_in
        layers: list[nn.Module] = [nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU()]
        for _ in range(max(0, n_layers - 2)):
            layers += [nn.Linear(hidden, hidden), nn.GELU()]
        layers += [nn.Linear(hidden, vocab_size)]
        self.net = nn.Sequential(*layers)
        # plain python float, NOT a buffer: state_dicts stay loadable with strict=True
        self.advance_bias = float(advance_bias)

    def logits(self, psi_bar: torch.Tensor) -> torch.Tensor:
        """psi_bar: (B, d_in) -> (B, vocab_size)."""
        return self.net(psi_bar)

    @staticmethod
    def _candidates(
        m_prev: torch.Tensor,
        plan_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Admissible slots K(m_prev) = {m_prev, min(m_prev+1, M-1)} and the end-of-plan mask."""
        m_next = torch.minimum(m_prev + 1, plan_len - 1)   # one-slot advance, capped at last slot
        cand = torch.stack([m_prev, m_next], dim=1)        # (B, 2)
        dup = cand[:, 0] == cand[:, 1]                     # end of plan: only "stay" is admissible
        return cand, dup

    @torch.no_grad()
    def step(
        self,
        psi_bar: torch.Tensor,      # (B, d_in)
        plan_op_ids: torch.Tensor,  # (B, M) op id at each plan slot
        m_prev: torch.Tensor,       # (B,) current pointer
        plan_len: torch.Tensor,     # (B,) number of real (non-pad) slots
    ):
        """One greedy decision step (argmax pointer update of Eq. 5).

        Returns ``(m_t, b_t, slot_logprob)`` where ``b_t = 1[m_t != m_prev]`` is the
        segment-boundary flag emitted for reward shaping (Eq. 8).
        """
        vocab_logits = self.logits(psi_bar)                       # (B, V)
        cand, dup = self._candidates(m_prev, plan_len)            # (B, 2)
        cand_ops = torch.gather(plan_op_ids, 1, cand)             # (B, 2) op ids at each slot
        cand_logits = torch.gather(vocab_logits, 1, cand_ops)     # (B, 2) logits at those ops
        cand_logits = cand_logits.clone()
        cand_logits[:, 1] += self.advance_bias                    # static offline calibration
        cand_logits[dup, 1] = float("-inf")                       # forbid non-admissible advance

        probs = F.softmax(cand_logits, dim=1)                     # (B, 2)
        choice = probs.argmax(dim=1)                              # 0=stay, 1=advance
        m_t = torch.gather(cand, 1, choice[:, None]).squeeze(1)
        b_t = (m_t != m_prev).long()
        slot_logprob = torch.log(
            torch.gather(probs, 1, choice[:, None]).squeeze(1) + 1e-9
        )
        return m_t, b_t, slot_logprob

    def slot_logprob(
        self,
        psi_bar: torch.Tensor,
        plan_op_ids: torch.Tensor,
        m_prev: torch.Tensor,
        plan_len: torch.Tensor,
        taken_slot: torch.Tensor,   # (B,) the slot actually executed
    ) -> torch.Tensor:
        """Differentiable log-prob of a taken slot, for the high-level policy gradient (Eq. 9)."""
        vocab_logits = self.logits(psi_bar)
        cand, dup = self._candidates(m_prev, plan_len)
        cand_ops = torch.gather(plan_op_ids, 1, cand)
        cand_logits = torch.gather(vocab_logits, 1, cand_ops)
        cand_logits = cand_logits + torch.tensor(
            [0.0, self.advance_bias], dtype=cand_logits.dtype, device=cand_logits.device
        )                                                          # same calibration as step()
        neg_inf_mask = torch.stack([torch.zeros_like(dup), dup], dim=1)
        cand_logits = cand_logits.masked_fill(neg_inf_mask, float("-inf"))
        logp = F.log_softmax(cand_logits, dim=1)                  # (B, 2)
        # 1 iff the pointer actually MOVED. Testing ``taken_slot == m_next`` instead is
        # wrong at the last slot, where m_next == m_prev and the advance candidate is
        # -inf-masked: a legitimate "stay" would then read as an advance and return -inf.
        is_advance = (taken_slot != m_prev).long()
        return torch.gather(logp, 1, is_advance[:, None]).squeeze(1)
