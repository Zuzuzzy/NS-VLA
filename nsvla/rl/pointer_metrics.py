"""Pointer-behaviour metrics for Stage-II RL.

These answer the question the online stage is judged on: does RL move *when* the
monotone pointer advances, relative to the physical event the advance is supposed to
wait for - the pick target actually being grasped?

Grasp criterion
---------------
The trace's ``probe['grasped']`` flag is not usable here. Its ``grip_gap < 0.03``
threshold is calibrated on thin rims and misses box-shaped objects entirely, missing
the large majority of provable grasps on some suites. The criterion used instead is
``obj_lift > 0.02 and grip_gap < 0.070``, whose recall against the simulator's own
grasp predicate is 1.000 on all four suites.

Which advance?
--------------
Keying on the *first* advance of an episode is correct when slot 0 is the ``pick``,
but wrong for a plan such as ``[open, pick, place_in]``, where the first advance
(open -> pick) legitimately precedes any grasp and would be scored premature by
construction. The primary statistic therefore keys on the advance **out of the pick
slot**; the first-advance variant is kept alongside it under ``*_firstadv``.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

# Recalibrated grasp criterion (see module docstring).
LIFT_TH = 0.02
HOLD_GAP = 0.070

_PICK_OPS = ("pick",)


def recal_grasped(probe: dict[str, float] | None) -> float:
    """1.0 iff this decision step shows a real grasp under the recalibrated criterion."""
    p = probe or {}
    lift = float(p.get("obj_lift", 0.0))
    gap = float(p.get("grip_gap", 9.0))
    return 1.0 if (lift > LIFT_TH and gap < HOLD_GAP) else 0.0


def _plan_ops(trace: Any) -> list[str]:
    return [str(p.get("op", "")) for p in (getattr(trace, "plan", None) or [])]


def episode_evidence(trace: Any) -> dict | None:
    """Per-episode pointer evidence from an in-memory ``EpisodeTrace``.

    Returns ``None`` for empty episodes. ``i_adv_pick`` is the index of the decision
    step at which the pointer first leaves the pick slot; ``i_grasp`` is the first
    decision step showing a recalibrated grasp.
    """
    steps = list(getattr(trace, "steps", []) or [])
    if not steps:
        return None
    ops = _plan_ops(trace)
    m = np.array([int(s.m_t) for s in steps], dtype=np.int64)
    b = np.array([int(s.b_t) for s in steps], dtype=np.int64)
    gr = np.array([recal_grasped(getattr(s, "probe", {})) for s in steps])

    i_grasp = int(np.flatnonzero(gr > 0.5)[0]) if np.any(gr > 0.5) else None
    i_adv_first = int(np.flatnonzero(b)[0]) if np.any(b) else None

    # first step whose pointer has moved PAST the pick slot
    pick_slot = next((i for i, op in enumerate(ops) if op in _PICK_OPS), None)
    i_adv_pick = None
    if pick_slot is not None:
        past = np.flatnonzero(m > pick_slot)
        if past.size:
            i_adv_pick = int(past[0])
    elif i_adv_first is not None:
        # no pick in the plan -> nothing to be premature about; leave None
        i_adv_pick = None

    return {
        "task_id": int(getattr(trace, "task_id", -1)) if str(getattr(trace, "task_id", "")).lstrip("-").isdigit() else getattr(trace, "task_id", None),
        "plan_len": len(ops),
        "has_pick": pick_slot is not None,
        "success": bool(getattr(trace, "success", False)),
        "n_adv": int(b.sum()),
        "n_decisions": len(steps),
        "final_m": int(m[-1]),
        "i_adv_first": i_adv_first,
        "i_adv_pick": i_adv_pick,
        "i_grasp": i_grasp,
    }


def _stats_for(recs: list[dict], adv_key: str) -> dict:
    """Premature-advance statistics keyed on ``adv_key`` (i_adv_pick or i_adv_first)."""
    advanced, never, lead_list, at_or_before = 0, 0, [], 0
    for r in recs:
        i_adv = r.get(adv_key)
        if i_adv is None:
            continue
        advanced += 1
        i_gr = r.get("i_grasp")
        if i_gr is None:
            never += 1
        elif i_gr <= i_adv:
            at_or_before += 1
        else:
            lead_list.append(i_gr - i_adv)
    before = never + len(lead_list)          # advance happened before a real grasp
    a = max(1, advanced)
    return {
        "n_episodes_with_advance": advanced,
        "adv_never_grasped": never,
        "adv_grasp_at_or_before": at_or_before,
        "adv_lead_gt0": len(lead_list),
        "frac_advance_before_grasp": (before / a) if advanced else None,
        # restricted to episodes that DID grasp: isolates pointer timing from grasp skill
        "frac_advance_before_grasp_grasped_only": (
            len(lead_list) / max(1, len(lead_list) + at_or_before)
            if (len(lead_list) + at_or_before) else None),
        "mean_lead_decisions": float(np.mean(lead_list)) if lead_list else None,
    }


def group_pointer_metrics(traces: Iterable[Any]) -> dict:
    """Pointer metrics over a group of rollouts (one Alg.1 iteration).

    Flat keys use the pick-slot-keyed advance; ``*_firstadv`` keys use the
    first-advance convention.
    """
    recs = [e for e in (episode_evidence(t) for t in traces) if e is not None]
    if not recs:
        return {"ptr_n_episodes": 0}
    main = _stats_for(recs, "i_adv_pick")
    first = _stats_for(recs, "i_adv_first")
    out = {"ptr_n_episodes": len(recs)}
    out.update({f"ptr_{k}": v for k, v in main.items()})
    out.update({f"ptr_{k}_firstadv": v for k, v in first.items()})
    out["ptr_frac_with_grasp"] = float(np.mean([r["i_grasp"] is not None for r in recs]))
    out["ptr_mean_advances"] = float(np.mean([r["n_adv"] for r in recs]))
    out["ptr_plan_len"] = int(recs[0]["plan_len"])
    out["ptr_M_class"] = "M1" if recs[0]["plan_len"] <= 1 else "Mge2"
    out["ptr_evidence"] = recs          # kept in the trace-side record, not metrics.jsonl
    return out
