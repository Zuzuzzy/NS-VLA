"""Trace -> metrics (NS-VLA §5.1). Traces are the only metric source.

Implements the paper's headline metrics off the symbolic traces:
  * **SR**  — success rate (fraction of episodes completing the task);
  * **PSCR** — Plan-Stage Completion Rate: fraction of plan primitives whose
    terminal precondition is met, approximated from boundary flags (segments
    entered) over plan length;
  * **Robustness** — ratio of LIBERO-Plus avg SR to LIBERO avg SR (percent);
  * segment statistics (mean segments/episode, mean segment length).

Pure functions over EpisodeTrace lists; no simulator involved.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from nsvla.eval.traces import EpisodeTrace


def success_rate(traces: list[EpisodeTrace]) -> float:
    if not traces:
        return 0.0
    return float(np.mean([1.0 if t.success else 0.0 for t in traces]))


def pscr(traces: list[EpisodeTrace]) -> float:
    """Plan-Stage Completion Rate: mean over episodes of (#segments entered / plan length)."""
    if not traces:
        return 0.0
    vals = []
    for t in traces:
        M = max(1, len(t.plan))
        entered = 1 + sum(int(s.b_t) for s in t.steps)   # segment count = 1 + #boundary fires
        vals.append(min(entered, M) / M)
    return float(np.mean(vals))


def segment_stats(traces: list[EpisodeTrace]) -> dict[str, float]:
    """Mean segments per episode and mean segment length (in decision steps)."""
    seg_counts, seg_lens = [], []
    for t in traces:
        n_seg = 1 + sum(int(s.b_t) for s in t.steps)
        seg_counts.append(n_seg)
        if n_seg > 0:
            seg_lens.append(len(t.steps) / n_seg)
    return {
        "mean_segments": float(np.mean(seg_counts)) if seg_counts else 0.0,
        "mean_segment_len": float(np.mean(seg_lens)) if seg_lens else 0.0,
    }


def sr_by_suite(traces: list[EpisodeTrace]) -> dict[str, float]:
    groups: dict[str, list[EpisodeTrace]] = defaultdict(list)
    for t in traces:
        groups[t.suite].append(t)
    return {suite: success_rate(ts) for suite, ts in groups.items()}


def robustness(libero_traces: list[EpisodeTrace], libero_plus_traces: list[EpisodeTrace]) -> float:
    """Robustness = 100 * avg SR(LIBERO-Plus) / avg SR(LIBERO)  (percent, paper §5.1)."""
    base = success_rate(libero_traces)
    if base <= 0.0:
        return 0.0
    return 100.0 * success_rate(libero_plus_traces) / base


def summarize(traces: list[EpisodeTrace]) -> dict[str, object]:
    """Metrics bundle for a trace set (written to ``<run_dir>/metrics.json``)."""
    out: dict[str, object] = {
        "n_episodes": len(traces),
        "SR": success_rate(traces),
        "PSCR": pscr(traces),
        "SR_by_suite": sr_by_suite(traces),
    }
    out.update(segment_stats(traces))
    return out
