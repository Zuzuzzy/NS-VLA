"""Primitive-conditioned visual-token sparsification -> c_t (NS-VLA App. F, Alg.1 line 15).

The active primitive ``u_hat_t`` selects a small salient subset of the visual
tokens ``psi_t`` to form the conditioned context ``c_t``. The paper's "top-K=32"
is implemented as a **proportional top-p** (keep ~12.5% of tokens, the same order
as K=32 out of 256), with:
  * **training**: a *soft* low-temperature gate (differentiable, all tokens kept
    but reweighted) so the selection gradient flows;
  * **inference**: a *hard* top-p mask (exact subset), the deployed behaviour.

Pure tensor ops. Selection scores are cosine relevance between a
per-primitive query and each token; the query is supplied by the caller (a
learned per-op embedding at train time), which keeps this module solver-agnostic.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def relevance_scores(tokens: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    """Cosine relevance of each visual token to the active-primitive query.

    tokens: (B, N, d)  query: (B, d)  ->  scores: (B, N) in [-1, 1].
    """
    t = F.normalize(tokens, dim=-1)
    q = F.normalize(query, dim=-1).unsqueeze(1)   # (B, 1, d)
    return (t * q).sum(dim=-1)                     # (B, N)


def _keep_count(n_tokens: int, top_p: float, k_floor: int) -> int:
    k = int(round(top_p * n_tokens))
    return max(k_floor, min(k, n_tokens))


def hard_topk_mask(scores: torch.Tensor, top_p: float, k_floor: int = 1) -> torch.Tensor:
    """Hard top-p selection mask (B, N) in {0,1}: keep the highest-scoring ~top_p tokens."""
    b, n = scores.shape
    k = _keep_count(n, top_p, k_floor)
    idx = scores.topk(k, dim=-1).indices              # (B, k)
    mask = torch.zeros_like(scores)
    mask.scatter_(1, idx, 1.0)
    return mask


def soft_gate(scores: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Soft per-token gate (B, N) in (0,1): sigmoid over the top-p relevance threshold.

    Tokens above the (1 - top_p) score quantile pass ~1, the rest are attenuated;
    a low temperature approaches the hard mask while staying differentiable.
    """
    b, n = scores.shape
    k = _keep_count(n, top_p, k_floor=1)
    # per-row threshold = the k-th largest score (kept tokens sit at/above it).
    thresh = scores.topk(k, dim=-1).values[:, -1:].detach()   # (B, 1)
    return torch.sigmoid((scores - thresh) / max(temperature, 1e-6))


def sparsify(
    tokens: torch.Tensor,           # (B, N, d) visual token features psi_t
    query: torch.Tensor,            # (B, d) active-primitive query
    top_p: float = 0.125,
    temperature: float = 0.5,
    training: bool = True,
    k_floor: int = 1,
) -> torch.Tensor:
    """Return the conditioned context ``c_t`` (B, N, d): tokens reweighted/selected by u_hat_t.

    Training -> soft gate (all tokens kept, reweighted). Inference -> hard top-p
    (unselected tokens zeroed). Downstream pooling/attention consumes ``c_t``.
    """
    scores = relevance_scores(tokens, query)          # (B, N)
    if training:
        gate = soft_gate(scores, temperature, top_p)   # (B, N)
    else:
        gate = hard_topk_mask(scores, top_p, k_floor)  # (B, N)
    return tokens * gate.unsqueeze(-1)


def derived_k(n_tokens: int, top_p: float = 0.125, k_floor: int = 1) -> int:
    """The absolute K implied by the proportional top-p (recorded for provenance)."""
    return _keep_count(n_tokens, top_p, k_floor)
