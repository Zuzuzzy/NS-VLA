"""NS-VLA: a neuro-symbolic vision-language-action framework.

Three components, wired in this order:

  * ``nsvla.encoder``  - the neuro-symbolic encoder. Extracts a fixed symbolic plan
    from the instruction once per episode (Eq. 4) and infers the active primitive at
    every decision step under a monotone {stay, +1} pointer (Eq. 5).
  * ``nsvla.solver``   - the action solver interface. Primitive-conditioned bridging
    and visual-token sparsification (Eq. 6, App. F) feed an ``ActionSolver``, which
    emits an H-step action chunk (Eq. 7). Any VLA or policy model can be plugged in;
    ``nsvla.solver.vla_adapter`` is the reference implementation.
  * ``nsvla.rl``       - online reinforcement learning. Bi-granular rewards (Eq. 8)
    drive a group-relative high-level update on the pointer and an advantage-weighted
    regression on the solver's low-level parameters (Eq. 9, Alg. 1).

Mechanism boundary, unchanged by any hyper-parameter:

  * the plan is extracted ONCE per episode (Eq. 4);
  * the pointer mask is {stay, +1} (Eq. 5), which gives Proposition 1;
  * bridging and sparsification are conditioned on the active primitive (Eq. 6, App. F);
  * control is H-step chunked (Eq. 7), which gives Proposition 2;
  * the reward is bi-granular with a stop-grad prototype potential (Eq. 8);
  * the update is high-level GRPO + low-level AWR with a KL anchor to BC (Eq. 9),
    which gives Proposition 3.

Importing this package stays light: torch, ZMQ and the simulator are imported lazily
inside the functions that need them.
"""
from __future__ import annotations

__version__ = "0.1.0"

from nsvla.primitives.vocab import (  # noqa: F401
    DEFAULT_OPS,
    NOOP_OP,
    PAD_OP,
    Plan,
    Primitive,
    PrimitiveVocab,
    default_vocab,
)

__all__ = [
    "__version__",
    "Primitive",
    "Plan",
    "PrimitiveVocab",
    "default_vocab",
    "DEFAULT_OPS",
    "PAD_OP",
    "NOOP_OP",
]
