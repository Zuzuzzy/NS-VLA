"""Per-primitive prototype shaping potentials (NS-VLA App. C, Alg.1 lines 16-17, 30-35).

A frozen shaping encoder ``E_w`` maps an observation to a latent ``ell_t``. Each
primitive sigma keeps a FIFO buffer ``B_sigma`` of successful-segment summaries
``ell_bar_sigma``; every ``U`` iterations the buffer is clustered into ``C``
prototypes ``{mu_{sigma,c}}`` (refreshed, **stop-grad**, treated as constant
within an update). The shaping potential is

    Phi_t = - min_c || ell_t - mu_{m_hat_t, c} ||^2

Prototypes feed **only the reward** (Eq. 8), never the policy gradient.

The buffer/cluster/potential logic here is pure tensor; the frozen encoder ``E_w``
is injected and GPU-side.
"""
from __future__ import annotations

from collections import deque

import torch


class PrototypeBank:
    """Per-primitive successful-segment buffers + cluster prototypes (stop-grad)."""

    def __init__(self, n_primitives: int, dim: int, buffer_cap: int = 64, n_clusters: int = 3):
        self.n_primitives = n_primitives
        self.dim = dim
        self.buffer_cap = buffer_cap
        self.n_clusters = n_clusters
        self._buffers: list[deque] = [deque(maxlen=buffer_cap) for _ in range(n_primitives)]
        # (n_primitives, C, dim); NaN rows mean "no prototype yet".
        self.prototypes = torch.full((n_primitives, n_clusters, dim), float("nan"))

    def add_segment(self, prim_id: int, seg_summary: torch.Tensor) -> None:
        """Insert a successful-segment summary ell_bar_sigma into buffer B_sigma (FIFO cap)."""
        self._buffers[prim_id].append(seg_summary.detach().flatten().float())

    def refresh(self) -> None:
        """Recompute prototypes {mu_{sigma,c}} from buffers via k-means (Alg.1 line 5)."""
        for sigma in range(self.n_primitives):
            buf = self._buffers[sigma]
            if len(buf) == 0:
                continue
            X = torch.stack(list(buf), dim=0)                    # (n, dim)
            centers = _kmeans(X, self.n_clusters)                # (<=C, dim)
            with torch.no_grad():
                self.prototypes[sigma] = float("nan")
                self.prototypes[sigma, : centers.shape[0]] = centers

    def potential(self, prim_id: int, ell_t: torch.Tensor) -> torch.Tensor:
        """Phi_t = - min_c || ell_t - mu_{sigma,c} ||^2  (0 if no prototype yet). stop-grad."""
        protos = self.prototypes[prim_id]                        # (C, dim)
        valid = ~torch.isnan(protos).any(dim=1)
        if valid.sum() == 0:
            return torch.zeros((), dtype=torch.float32)
        with torch.no_grad():
            d2 = ((protos[valid] - ell_t.detach().flatten().float()) ** 2).sum(dim=1)
            return -d2.min()


    def potential_np(self, prim_id: int, ell_t) -> float:
        """numpy-side convenience: Phi_t as a plain float (reward path only)."""
        import numpy as np

        t = torch.as_tensor(np.asarray(ell_t, dtype="float32"))
        return float(self.potential(prim_id, t))

    def n_filled(self) -> list[int]:
        """Per-primitive buffer occupancy (logged each iteration)."""
        return [len(b) for b in self._buffers]

    def state_dict(self) -> dict:
        return {
            "n_primitives": self.n_primitives,
            "dim": self.dim,
            "buffer_cap": self.buffer_cap,
            "n_clusters": self.n_clusters,
            "buffers": [torch.stack(list(b), 0) if len(b) else torch.zeros(0, self.dim)
                        for b in self._buffers],
            "prototypes": self.prototypes,
        }

    def load_state_dict(self, d: dict) -> None:
        self._buffers = [deque(maxlen=self.buffer_cap) for _ in range(self.n_primitives)]
        for i, X in enumerate(d.get("buffers", [])):
            if i >= self.n_primitives:
                break
            for row in X:
                self._buffers[i].append(row.detach().float())
        if "prototypes" in d:
            self.prototypes = d["prototypes"]


def _kmeans(X: torch.Tensor, k: int, iters: int = 20) -> torch.Tensor:
    """Tiny deterministic k-means (>= returns <=k centers). CPU, no sklearn dep."""
    n = X.shape[0]
    k = min(k, n)
    centers = X[:k].clone()                                      # deterministic init
    for _ in range(iters):
        d2 = torch.cdist(X, centers) ** 2                        # (n, k)
        assign = d2.argmin(dim=1)                                # (n,)
        new = centers.clone()
        for c in range(k):
            mask = assign == c
            if mask.any():
                new[c] = X[mask].mean(dim=0)
        if torch.allclose(new, centers):
            centers = new
            break
        centers = new
    return centers
