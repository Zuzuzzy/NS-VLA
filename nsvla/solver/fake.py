"""Deterministic CPU stand-in solver.

The chunk is a pure function of ``(image, instruction, proprio)`` - no weights, no
randomness - so the harness can be exercised end to end without loading a policy, and
the Proposition 2 equivalence is exact: with M = 1 the rendered sub-instruction equals
the full instruction, so this solver returns bit-identical chunks on the NS-VLA path
and the bare-controller path.
"""
from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

from nsvla.solver.base import ActionSolver, register_solver


@register_solver("fake")
class FakeSolver(ActionSolver):
    """Hash-based stand-in with the exact ``act`` contract."""

    def __init__(self, cfg: Any = None, chunk_H: int = 8, action_dim: int = 7):
        if cfg is not None:
            chunk_H = getattr(cfg, "chunk_H", chunk_H)
            action_dim = getattr(cfg, "action_dim", action_dim)
        self.chunk_H = chunk_H
        self.action_dim = action_dim

    def act(
        self,
        image: np.ndarray,
        instruction: str,
        proprio: np.ndarray,
        wrist_image: np.ndarray | None = None,
    ) -> np.ndarray:
        key = hashlib.sha256()
        key.update(np.ascontiguousarray(image).tobytes())
        key.update(instruction.encode("utf-8"))
        key.update(np.ascontiguousarray(np.asarray(proprio, dtype=np.float32)).tobytes())
        seed = int.from_bytes(key.digest()[:8], "little")
        rng = np.random.default_rng(seed)
        chunk = rng.uniform(-1.0, 1.0, size=(self.chunk_H, self.action_dim))
        return chunk.astype(np.float32)
