"""Symbolic primitive vocabulary and plan types (NS-VLA §4.1, Eq. 2 & 4).

Dependency-free on purpose (no torch), so ``import nsvla`` stays light and the
symbolic layer can be reasoned about without a GPU.

A *primitive* is an (op, object, support) triple: ``op`` names the symbolic
operation drawn from the shared vocabulary ``U``; ``object`` is the manipulated
referent; ``support`` is the target referent (destination / container / anchor).
A *plan* ``p = (u^(1), ..., u^(M))`` is fixed for the whole episode; slots beyond
the real length ``M`` are right-padded with ``<pad>`` which the monotone pointer
ignores (Eq. 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Right-pad slot: the monotone pointer never dwells here (Eq. 4/5).
PAD_OP = "<pad>"
# Explicit no-op primitive (e.g. idle / settle), kept distinct from padding.
NOOP_OP = "<noop>"

# The eight manipulation operations LIBERO instructions decompose into (Fig. 4a).
DEFAULT_OPS: list[str] = [
    "pick",       # grasp / pick up an object
    "place_on",   # put object on a support surface
    "place_in",   # put object inside a container
    "place_rel",  # put object at a spatial relation to an anchor (left/right/front/back of)
    "open",       # open a drawer / door / articulated part
    "close",      # close a drawer / door / lid
    "turn_on",    # actuate a switch / stove
    "push_to",    # slide / push an object to a location
]


@dataclass
class Primitive:
    """A single symbolic operation with its argument structure (op, object, support)."""

    op: str
    object: str | None = None
    support: str | None = None

    @property
    def args(self) -> dict[str, Any]:
        d: dict[str, Any] = {}
        if self.object is not None:
            d["object"] = self.object
        if self.support is not None:
            d["support"] = self.support
        return d

    def is_real(self) -> bool:
        return self.op not in (PAD_OP, NOOP_OP)

    def key(self) -> str:
        obj = self.object or ""
        sup = self.support or ""
        inner = obj + ("," + sup if sup else "")
        return f"{self.op}({inner})"

    def to_dict(self) -> dict[str, Any]:
        return {"op": self.op, "args": self.args}


@dataclass
class Plan:
    """Episode-level plan, fixed across the episode (Eq. 2). Right-padded to ``max_len``."""

    primitives: list[Primitive]
    max_len: int = 6

    @property
    def M(self) -> int:
        """Number of real (non-pad, non-noop) primitives = plan length."""
        return sum(1 for p in self.primitives if p.is_real())

    def real(self) -> list[Primitive]:
        return [p for p in self.primitives if p.is_real()]

    def padded(self) -> list[Primitive]:
        """Right-pad to ``max_len`` with ``<pad>`` primitives."""
        prims = list(self.primitives[: self.max_len])
        prims += [Primitive(PAD_OP)] * (self.max_len - len(prims))
        return prims

    def padded_op_ids(self, vocab: "PrimitiveVocab") -> list[int]:
        return [vocab.id(p.op) for p in self.padded()]

    def to_list(self) -> list[dict[str, Any]]:
        return [p.to_dict() for p in self.real()]


class PrimitiveVocab:
    """Bidirectional op<->id map. ``<pad>`` and ``<noop>`` are always present."""

    def __init__(self, ops: list[str]):
        ops = list(ops)
        for special in (PAD_OP, NOOP_OP):
            if special not in ops:
                ops.append(special)
        self._ops = ops
        self._id = {o: i for i, o in enumerate(ops)}

    def id(self, op: str) -> int:
        return self._id[op]

    def op(self, i: int) -> str:
        return self._ops[i]

    def has(self, op: str) -> bool:
        return op in self._id

    @property
    def ops(self) -> list[str]:
        return list(self._ops)

    def __len__(self) -> int:
        return len(self._ops)

    def __contains__(self, op: str) -> bool:
        return op in self._id


def default_vocab() -> PrimitiveVocab:
    return PrimitiveVocab(DEFAULT_OPS)
