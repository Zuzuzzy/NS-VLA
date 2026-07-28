"""Stage I-b: 1-shot BC warm-start of the low-level policy (App. C.2).

Behaviour cloning on the single demonstration per task (1-shot main setting):
chunked L2 regression of pi^l_theta onto the demo action chunks, conditioned via
the primitive bridge (``nsvla.solver.bridge``). Selects the held-out-best
checkpoint and freezes it as the BC anchor pi_BC that the Stage-II KL term
(``nsvla.rl.grpo_awr``) is anchored to.
"""
from __future__ import annotations

from nsvla.config import Config


def train_bc(cfg: Config) -> None:
    """1-shot chunked L2 BC of the low-level policy; save the pi_BC anchor.

    Steps:
      1. Repack the 1-shot demo into (obs, primitive, proprio) -> action-chunk pairs.
      2. Condition the solver via solver.bridge (rendered sub-instruction primary).
      3. Chunked L2 regression || pi^l_theta(e_t) - A_t ||_2 over H=8 steps.
      4. Select held-out-best; freeze as pi_BC (KL anchor for Stage II).
    """
    raise NotImplementedError(
        "Stage I-b BC warm-start runs inside the action solver's own environment, "
        "because it updates the solver's parameters rather than anything in this "
        "package; see nsvla/solver/vla_adapter.py for the reference solver."
    )
