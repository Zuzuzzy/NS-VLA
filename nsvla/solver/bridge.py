"""Primitive-conditioned bridging  e_t  /  rendered sub-instruction  x_tilde_t  (Eq. 6).

Two configurable forms:
  * **embed**   : e_t = Wu * Embed(u_hat_t, arg_t) + Wpsi * psi_bar_t + WS * S_t
                  (Eq. 6) — a learned projection fusing the symbolic op-arg pair,
                  the pooled context, and proprioception onto the controller dim.
  * **render**  : replace the embedding stream with a natural-language sub-instruction
                  x_tilde_t, used when the action solver consumes language (the
                  VLA-Adapter reference solver does). This is the default bridging;
                  ``embed`` is kept as an ablation for solvers that expose an
                  embedding input.

**Proposition 2 hook**: when M = 1 the single primitive spans the whole task, so
``render_sub_instruction`` returns the original instruction verbatim -> the solver
receives identical inputs to the bare controller -> bit-exact trajectory
equivalence.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from nsvla.primitives.vocab import Plan, Primitive, PrimitiveVocab

# Natural-language templates per op for the rendered sub-instruction.
_RENDER_TEMPLATES = {
    "pick": "pick up the {object}",
    "place_on": "put the {object} on the {support}",
    "place_in": "put the {object} in the {support}",
    "place_rel": "put the {object} next to the {support}",
    "open": "open the {object}",
    "close": "close the {object}",
    "turn_on": "turn on the {object}",
    "push_to": "push the {object} to the {support}",
}


def render_sub_instruction(plan: Plan, m_t: int, instruction: str) -> str:
    """Render the active primitive u_hat_t = plan[m_t] into a focused sub-instruction.

    When the plan is a single primitive (M == 1) the sub-instruction IS the whole
    task, so we return ``instruction`` verbatim — this is what makes the M=1
    solver inputs identical to those of the bare controller (Proposition 2).
    """
    reals = plan.real()
    if len(reals) <= 1:
        return instruction
    m_t = max(0, min(m_t, len(reals) - 1))
    prim = reals[m_t]
    tmpl = _RENDER_TEMPLATES.get(prim.op, "{object}")
    obj = prim.object or "object"
    sup = prim.support or "target"
    return tmpl.format(object=obj, support=sup)


class PrimitiveBridge(nn.Module):
    """Embed-mode bridging (Eq. 6): fuse op-arg embedding + context + proprio -> e_t."""

    def __init__(
        self,
        vocab_size: int,
        d_psi: int,
        d_state: int,
        d_embed: int = 512,
        d_controller: int = 512,
        n_arg_buckets: int = 512,
    ):
        super().__init__()
        # Embed(u_hat, arg): op table + a hashed-argument table, summed -> R^{d_embed}.
        self.op_embed = nn.Embedding(vocab_size, d_embed)
        self.arg_embed = nn.Embedding(n_arg_buckets, d_embed)
        self.n_arg_buckets = n_arg_buckets
        self.W_u = nn.Linear(d_embed, d_controller, bias=False)     # Wu
        self.W_psi = nn.Linear(d_psi, d_controller, bias=False)     # Wpsi
        self.W_s = nn.Linear(d_state, d_controller, bias=False)     # WS

    def arg_bucket(self, arg_text: str | None) -> int:
        if not arg_text:
            return 0
        return (hash(arg_text) % (self.n_arg_buckets - 1)) + 1

    def forward(
        self,
        op_ids: torch.Tensor,        # (B,) executed op id u_hat_t
        arg_ids: torch.Tensor,       # (B,) hashed argument bucket
        psi_bar: torch.Tensor,       # (B, d_psi) pooled context
        state: torch.Tensor,         # (B, d_state) proprio S_t
    ) -> torch.Tensor:
        """Return e_t in R^{d_controller} (Eq. 6)."""
        op_arg = self.op_embed(op_ids) + self.arg_embed(arg_ids)   # Embed(u_hat, arg)
        return self.W_u(op_arg) + self.W_psi(psi_bar) + self.W_s(state)


def op_arg_ids(
    plan: Plan, m_t: int, vocab: PrimitiveVocab, bridge: PrimitiveBridge
) -> tuple[int, int]:
    """Convenience: (op_id, arg_bucket) for the active primitive at pointer m_t."""
    reals = plan.real()
    m_t = max(0, min(m_t, len(reals) - 1)) if reals else 0
    prim = reals[m_t] if reals else Primitive(vocab.op(0))
    return vocab.id(prim.op), bridge.arg_bucket(prim.object)
