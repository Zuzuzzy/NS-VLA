"""Run-directory naming helpers.

The run seed is carried by the run directory name's ``_s<k>`` suffix and nowhere else:
``EpisodeTrace.seed`` is the per-episode initial-state index, a different quantity.
Multi-seed aggregation therefore reads the seed from the directory name, which is why
``seed_of`` raises instead of returning ``None`` - a run whose name does not conform
must fail loudly rather than silently disappear from an aggregate table.
"""
from __future__ import annotations


def seed_of(run_id: str) -> int:
    """Run seed from the ``_s<k>`` suffix of a run id (``spatial_nsvla_s2`` -> 2)."""
    tail = run_id.rsplit("_s", 1)[-1]
    if not tail.isdigit() or "_s" not in run_id:
        raise ValueError(
            f"cannot parse a run seed from run_id {run_id!r}: expected a trailing "
            f"'_s<digits>' suffix (e.g. 'spatial_nsvla_s2')."
        )
    return int(tail)


def run_id(suite: str, arm: str, seed: int) -> str:
    """Canonical run id for one (suite, arm, seed) cell."""
    return f"{suite}_{arm}_s{int(seed)}"
