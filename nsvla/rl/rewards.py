"""Bi-granular reward streams (NS-VLA §4.3, Eq. 8).

Granularity-matched (paper lines 147-153):
  * **high-level** (per-decision primitive selection), sparse milestone return:
        R^h(tau) = r_task + lambda_seg * sum_t b_t
  * **low-level** (per-chunk control), dense per-chunk shaping return:
        r^l_t = lambda_prog (gamma * Phi_{t+1} - Phi_t)
              + lambda_sub  * r^sub_t
              + lambda_sm   * r^sm_t
              + lambda_gr   * r^gr_t
where ``Phi_t`` is the prototype shaping potential (rl/prototypes.py), ``r^sub``
a sub-goal reaching indicator, ``r^sm`` an action-smoothness penalty, ``r^gr`` a
gripper-alignment term. Weights are running-std normalized to a common scale then
scaled by milestone:progress:aux = 1:0.5:0.1.

Pure numpy/tensor, with no environment coupling. The telescoping identity of the
progress term is **Proposition 3**.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def telescoped_progress_sum(phis: np.ndarray, gamma: float = 1.0) -> float:
    """Sum_t (gamma * Phi_{t+1} - Phi_t) over a trajectory of potentials ``phis``.

    Proposition 3 (telescoping): with gamma == 1 this collapses to
    ``Phi_T - Phi_0``; with a terminal Phi_T == 0 it equals ``-Phi_0 = -Phi(s0)``,
    so the dense shaping adds no new optimum (potential-based shaping is policy
    invariant, App. B.3).
    """
    phis = np.asarray(phis, dtype=np.float64)
    return float(np.sum(gamma * phis[1:] - phis[:-1]))


def progress_rewards(phis: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """Per-step progress reward  gamma * Phi_{t+1} - Phi_t  (length T-1)."""
    phis = np.asarray(phis, dtype=np.float64)
    return gamma * phis[1:] - phis[:-1]


def high_level_return(
    r_task: float, boundary_flags: np.ndarray, lambda_seg: float = 1.0
) -> float:
    """R^h = r_task + lambda_seg * sum_t b_t  (Eq. 8, high level)."""
    b = np.asarray(boundary_flags, dtype=np.float64)
    return float(r_task) + lambda_seg * float(b.sum())


def action_smoothness(actions: np.ndarray) -> np.ndarray:
    """r^sm proxy: per-step squared change ||a_{t+1} - a_t||^2 (to be *penalized*)."""
    a = np.asarray(actions, dtype=np.float64)
    if a.shape[0] < 2:
        return np.zeros((0,), dtype=np.float64)
    return np.sum((a[1:] - a[:-1]) ** 2, axis=-1)


@dataclass
class RunningStd:
    """Welford running mean/std for online reward-component normalization.

    ``scale`` (divide by std, do NOT subtract the mean) is what the reward assembler
    uses: subtracting a per-step constant from the progress term would break the
    telescoping identity of Proposition 3 (a potential difference must stay a
    potential difference), so only the *scale* of each component is equalized.
    """

    count: float = 0.0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: np.ndarray) -> None:
        for v in np.asarray(x, dtype=np.float64).reshape(-1):
            self.count += 1.0
            delta = v - self.mean
            self.mean += delta / self.count
            self.m2 += delta * (v - self.mean)

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return float(np.sqrt(self.m2 / (self.count - 1)) + 1e-8)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.mean) / self.std

    def scale(self, x: np.ndarray) -> np.ndarray:
        """Scale-only normalization x / sigma (mean kept — see class docstring)."""
        return np.asarray(x, dtype=np.float64) / self.std

    def to_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean, "m2": self.m2}

    @classmethod
    def from_dict(cls, d: dict) -> "RunningStd":
        return cls(count=float(d["count"]), mean=float(d["mean"]), m2=float(d["m2"]))


# --------------------------------------------------------------------------- #
# sub-goal / gripper-alignment indicators (Eq. 8 aux terms)
# --------------------------------------------------------------------------- #
# Ops during which the gripper is expected to be HOLDING the object. `pick` is
# excluded because the approach phase of a pick is legitimately open; a pick's
# sub-goal is the grasp itself, which is the r^sub term.
_CARRY_OPS = ("place_on", "place_in", "place_rel", "push_to")


def subgoal_indicator(op: str, grasped: float) -> float:
    """r^sub_t: 1 when the active primitive's sub-goal is currently satisfied.

    The only sub-goal state observable at a chunk boundary without extra sensing is
    the pick-target grasp state (``probe['grasped']`` = gripper closed AND target
    lifted off its rest height; see eval.run_suite._probe_env, which the RL rollout
    reuses verbatim). For a ``pick`` the sub-goal IS the grasp; for a transport /
    placement op the object must still be held; other ops are unconstrained (0).
    """
    g = float(grasped)
    if op == "pick":
        return g
    if op in _CARRY_OPS:
        return g
    return 0.0


def gripper_alignment(op: str, grip_gap: float, closed_thresh: float = 0.03) -> float:
    """r^gr_t: 1 when the gripper aperture agrees with what the active primitive needs.

    ``pick``/carry ops want a closing gripper, ``open``/``close``/``turn_on`` (articulated
    interaction) want it open. Purely a function of the symbolic op and the measured
    finger gap - no learned part, no policy input (aux weight 0.1).
    """
    closed = float(grip_gap) < closed_thresh
    if op in _CARRY_OPS or op == "pick":
        return 1.0 if closed else 0.0
    return 0.0 if closed else 1.0


def low_level_reward(
    phis: np.ndarray,
    r_sub: np.ndarray,
    r_sm: np.ndarray,
    r_gr: np.ndarray,
    gamma: float = 0.99,
    lambda_prog: float = 1.0,
    lambda_sub: float = 0.5,
    lambda_sm: float = 0.1,
    lambda_gr: float = 0.1,
) -> np.ndarray:
    """Assemble the dense per-chunk low-level reward r^l_t (Eq. 8, low level).

    Lengths align to T-1 progress steps; ``r_sm`` enters with a negative sign
    (smoothness penalty). All arrays are per-decision-step, already normalized by
    the caller if ``normalize_running_std`` is on.
    """
    prog = progress_rewards(phis, gamma)             # (T-1,)
    T1 = prog.shape[0]

    def _fit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if x.shape[0] >= T1:
            return x[:T1]
        return np.pad(x, (0, T1 - x.shape[0]))

    return (
        lambda_prog * prog
        + lambda_sub * _fit(r_sub)
        - lambda_sm * _fit(r_sm)
        + lambda_gr * _fit(r_gr)
    )


# --------------------------------------------------------------------------- #
# rollout-level assembler (Alg.1 lines 20-23)
# --------------------------------------------------------------------------- #
class RewardShaper:
    """Assemble both reward streams of Eq. 8 for a rollout, with running-std scaling.

    One shaper instance lives for the whole Stage-II run (its running statistics are
    part of the checkpoint), so component scales are equalized *across* iterations:
    each component is first brought to a common magnitude by its running std, and only
    then are the small priority coefficients applied.

    Per rollout, given ``T`` decision steps:
      * ``phis``    : (T+1,) shaping potentials Phi_0..Phi_T (Phi_T = 0 at an absorbing
        terminal, App. C.1 "Early termination and absorbing terminals");
      * ``b_flags`` : (T,) boundary indicators b_t;
      * ``r_task``  : terminal success indicator in {0,1};
      * ``r_sub`` / ``r_gr`` : (T,) indicators from ``subgoal_indicator`` / ``gripper_alignment``;
      * ``actions`` : (T, H, action_dim) executed chunks -> the smoothness penalty.

    Returns ``{R_high, r_low (T,), terms {...}}``. Nothing here touches the policy;
    the trace is the only consumer, so every number is reproducible offline.
    """

    COMPONENTS = ("prog", "sub", "sm", "gr")

    def __init__(self, cfg=None):
        from nsvla.config import RewardConfig

        self.cfg = cfg or RewardConfig()
        self.stats: dict[str, RunningStd] = {c: RunningStd() for c in self.COMPONENTS}

    # ---------------------------------------------------------------- #
    def episode(
        self,
        phis: np.ndarray,
        b_flags: np.ndarray,
        r_task: float,
        r_sub: np.ndarray,
        r_gr: np.ndarray,
        actions: np.ndarray | None = None,
        update_stats: bool = True,
    ) -> dict:
        cfg = self.cfg
        phis = np.asarray(phis, dtype=np.float64).reshape(-1)
        b = np.asarray(b_flags, dtype=np.float64).reshape(-1)
        T = b.shape[0]

        # raw components, all length T
        prog_raw = np.zeros(T, dtype=np.float64)
        if phis.shape[0] >= 2:
            p = progress_rewards(phis, cfg.gamma)
            prog_raw[: min(T, p.shape[0])] = p[:T]
        sub_raw = _fit_len(r_sub, T)
        gr_raw = _fit_len(r_gr, T)
        if actions is None:
            sm_raw = np.zeros(T, dtype=np.float64)
        else:
            a = np.asarray(actions, dtype=np.float64).reshape(T, -1)
            sm_raw = np.concatenate([[0.0], np.sum(np.diff(a, axis=0) ** 2, axis=-1)])[:T]

        raw = {"prog": prog_raw, "sub": sub_raw, "sm": sm_raw, "gr": gr_raw}
        if update_stats:
            for c, v in raw.items():
                self.stats[c].update(v)
        if cfg.normalize_running_std:
            nrm = {c: self.stats[c].scale(v) for c, v in raw.items()}
        else:
            nrm = dict(raw)

        r_low = (
            cfg.lambda_prog * nrm["prog"]
            + cfg.lambda_sub * nrm["sub"]
            - cfg.lambda_sm * nrm["sm"]
            + cfg.lambda_gr * nrm["gr"]
        )
        R_high = high_level_return(r_task, b, cfg.lambda_seg)
        return {
            "R_high": float(R_high),
            "r_low": r_low,
            "raw": raw,
            "norm": nrm,
        }

    # ---------------------------------------------------------------- #
    def state_dict(self) -> dict:
        return {c: s.to_dict() for c, s in self.stats.items()}

    def load_state_dict(self, d: dict) -> None:
        for c, sd in d.items():
            if c in self.stats:
                self.stats[c] = RunningStd.from_dict(sd)


def _fit_len(x, T: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape[0] >= T:
        return x[:T]
    return np.pad(x, (0, T - x.shape[0]))
