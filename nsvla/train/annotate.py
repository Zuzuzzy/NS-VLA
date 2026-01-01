"""Offline demo annotation: frames -> per-frame primitive segments (NS-VLA App. C.2).

Turns a 1-shot demonstration into ordered primitive **segments** (one per plan
slot) plus a segment-end **window** ``w`` of frames used as positive labels for
the monotone-pointer classifier (Stage I-a). Segment boundaries are derived from
cheap, interpretable signals obtained by replaying the demo's mujoco ``states``
through the LIBERO env (SPEC task 3):
  * gripper open/close events (grasp / release transitions), and
  * articulated-joint plateaus for ``open`` / ``close`` / ``turn_on`` (drawer /
    door / stove-knob qpos), and
  * object-motion settle for ``push_to``.

The annotation is **plan-guided**: the plan (ordered primitives from the plan
extractor) fixes the number of segments ``M`` and their order; the signals only
locate the ``M-1`` internal boundaries. This makes the hard invariants — segment
order == plan order, no overlap, full cover, #segments == M — hold by construction.

Two layers:
  * a **pure segmentation core** (``segments_from_boundaries``, ``detect_gripper_events``,
    ``segment_invariants_ok``) — deterministic and simulator-free;
  * the **replay annotator** (``annotate_demo``) that steps a LIBERO env through the
    hdf5 states, reads signals, and locates the plan-guided boundaries.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# op -> completion-signal family
GRASP_OPS = {"pick"}
RELEASE_OPS = {"place_on", "place_in", "place_rel"}
JOINT_OPS = {"open", "close", "turn_on"}
SETTLE_OPS = {"push_to"}

# joint-name keywords used to match a plan object/support to a mujoco joint.
_JOINT_KEYWORDS = (
    "drawer", "cabinet", "stove", "button", "microwave",
    "door", "knob", "hinge", "slide", "faucet", "handle",
)


@dataclass
class Segment:
    """One primitive segment over frame interval [start, end) with a label window."""

    op: str
    start: int
    end: int          # exclusive
    window: list[int] = field(default_factory=list)  # segment-end frames used as clf labels
    signal: str = ""  # which signal located this segment's end boundary

    @property
    def length(self) -> int:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "op": self.op, "start": self.start, "end": self.end,
            "window": list(self.window), "signal": self.signal,
        }


# --------------------------------------------------------------------------- #
# pure segmentation core
# --------------------------------------------------------------------------- #
def detect_gripper_events(gripper: np.ndarray, thresh: float = 0.5) -> list[int]:
    """Frame indices where the gripper state crosses (open<->closed). ``gripper`` in [0,1]."""
    g = (np.asarray(gripper, dtype=np.float64) > thresh).astype(np.int64)
    if g.shape[0] < 2:
        return []
    changes = np.nonzero(g[1:] != g[:-1])[0] + 1
    return changes.tolist()


def detect_approach_boundaries(
    ee_target_dist: np.ndarray, contact_thresh: float = 0.05
) -> list[int]:
    """Frames where the end-effector first reaches a target (distance crosses below thresh)."""
    d = np.asarray(ee_target_dist, dtype=np.float64)
    if d.shape[0] < 2:
        return []
    near = (d < contact_thresh).astype(np.int64)
    onsets = np.nonzero((near[1:] == 1) & (near[:-1] == 0))[0] + 1
    return onsets.tolist()


def merge_boundaries(T: int, *boundary_lists: list[int], min_gap: int = 1) -> list[int]:
    """Union boundary indices, drop those at 0/T, dedupe, enforce a minimum segment gap."""
    cand = sorted({b for bl in boundary_lists for b in bl if 0 < b < T})
    out: list[int] = []
    for b in cand:
        if out and b - out[-1] < min_gap:
            continue
        out.append(b)
    return out


def segments_from_boundaries(
    T: int, boundaries: list[int], ops: list[str] | None = None, window: int = 3,
    signals: list[str] | None = None,
) -> list[Segment]:
    """Partition [0, T) at ``boundaries`` into contiguous, ordered, non-overlapping segments.

    Guarantees (checked by ``segment_invariants_ok``):
      * segments are ordered and non-overlapping,
      * they cover every frame in [0, T) with no gaps,
      * each segment's label window is its last ``w`` frames.
    ``ops`` (optional) labels segments in order; extra/missing are padded/ignored.
    """
    bounds = [0] + merge_boundaries(T, boundaries) + [T]
    segs: list[Segment] = []
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        if e <= s:
            continue
        op = ops[i] if ops and i < len(ops) else "<unlabeled>"
        sig = signals[i] if signals and i < len(signals) else ""
        w = list(range(max(s, e - window), e))
        segs.append(Segment(op=op, start=s, end=e, window=w, signal=sig))
    return segs


def segment_invariants_ok(segments: list[Segment], T: int) -> bool:
    """True iff segments are ordered, non-overlapping, and exactly cover [0, T)."""
    if not segments:
        return T == 0
    if segments[0].start != 0 or segments[-1].end != T:
        return False
    for i, seg in enumerate(segments):
        if seg.end <= seg.start:
            return False
        if i > 0 and seg.start != segments[i - 1].end:
            return False
        if any(f < seg.start or f >= seg.end for f in seg.window):
            return False
    return True


# --------------------------------------------------------------------------- #
# signal extraction (LIBERO replay)
# --------------------------------------------------------------------------- #
def gripper_closed_from_command(actions: np.ndarray) -> np.ndarray:
    """Boolean per-frame 'gripper closing' from the demo's gripper command.

    ``actions`` is (T, 7); the last dim is the gripper command (-1 open, +1 close).
    This is the robust grasp/release signal: unlike finger separation it is a clean
    square wave and does not fail on wide objects that never fully close the fingers.
    """
    a = np.asarray(actions, dtype=np.float64)
    return (a[:, -1] > 0).astype(np.int64)


def _runs(arr: np.ndarray) -> list[tuple[int, int, int]]:
    """Contiguous runs of equal value as (start, end_exclusive, value)."""
    a = np.asarray(arr)
    if a.size == 0:
        return []
    cuts = np.nonzero(a[1:] != a[:-1])[0] + 1
    bounds = [0, *cuts.tolist(), a.size]
    return [(bounds[i], bounds[i + 1], int(a[bounds[i]])) for i in range(len(bounds) - 1)]


def debounce_gripper(
    closed: np.ndarray,
    min_hold: int = 12,
    object_traces: dict[str, np.ndarray] | None = None,
    move_eps: float = 0.01,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Suppress spurious short grasp blips from approach-phase jitter.

    A closed run shorter than ``min_hold`` frames during which **no tracked object
    displaces by more than ``move_eps``** is not a real grasp (the gripper twitched
    while approaching) and is rewritten to 'open'. Short closed runs that *do* move
    an object are kept, so genuinely brief transports survive. Returns the cleaned
    trace and a list of the suppressed runs (for the annotation audit trail).
    """
    out = np.asarray(closed).copy()
    suppressed: list[dict[str, Any]] = []
    for s, e, v in _runs(out):
        if v != 1 or (e - s) >= min_hold:
            continue
        moved = 0.0
        for pos in (object_traces or {}).values():
            span = pos[s:e]
            if span.shape[0] >= 2:
                moved = max(moved, float(np.linalg.norm(span - span[0], axis=1).max()))
        if moved <= move_eps:
            out[s:e] = 0
            suppressed.append({"start": int(s), "end": int(e),
                               "len": int(e - s), "max_obj_disp": round(moved, 4)})
    return out, suppressed


def gripper_closed_trace(gripper_states: np.ndarray, open_frac: float = 0.6) -> np.ndarray:
    """Boolean per-frame 'gripper closed' from finger separation (physical, reference).

    ``gripper_states`` is (T, 2) finger joint positions; separation = g0 - g1.
    Closed when separation drops below ``open_frac`` * (max separation). Adaptive,
    but can miss wide-object grasps — the command trace above is preferred for
    boundary location; this remains for physical cross-checking / tests.
    """
    gs = np.asarray(gripper_states, dtype=np.float64)
    sep = gs[:, 0] - gs[:, 1]
    thr = open_frac * float(np.max(sep)) if sep.size else 0.0
    return (sep < thr).astype(np.int64)


def replay_joint_traces(env, states: np.ndarray, keywords=_JOINT_KEYWORDS) -> dict[str, np.ndarray]:
    """Replay ``states`` through ``env`` and return per-frame qpos for articulated joints.

    Only joints whose name contains one of ``keywords`` are tracked (drawer / door /
    stove-knob / microwave). No rendering — just ``set_state`` + ``sim.forward``.
    """
    sim = env.env.sim
    jnames = [sim.model.joint_id2name(j) for j in range(sim.model.njnt)]
    cand = [jn for jn in jnames if jn and any(k in jn.lower() for k in keywords)]
    traces = {jn: np.empty(len(states), dtype=np.float64) for jn in cand}
    for t in range(len(states)):
        env.set_state(states[t])
        sim.forward()
        for jn in cand:
            traces[jn][t] = float(np.ravel(sim.data.get_joint_qpos(jn))[0])
    return traces


def replay_object_traces(env, states: np.ndarray, obj_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Replay and return per-frame body xpos (T,3) for the given object names."""
    sim = env.env.sim
    bid = env.env.obj_body_id
    names = [n for n in obj_names if n in bid]
    tr = {n: np.empty((len(states), 3), dtype=np.float64) for n in names}
    for t in range(len(states)):
        env.set_state(states[t])
        sim.forward()
        for n in names:
            tr[n][t] = np.array(sim.data.body_xpos[bid[n]])
    return tr


# --------------------------------------------------------------------------- #
# per-op completion detectors
# --------------------------------------------------------------------------- #
def _next_gripper_transition(closed: np.ndarray, cursor: int, to_closed: bool) -> int | None:
    """First frame > cursor where gripper transitions to closed (grasp) / open (release)."""
    for t in range(max(cursor + 1, 1), len(closed)):
        if to_closed and closed[t] == 1 and closed[t - 1] == 0:
            return t
        if (not to_closed) and closed[t] == 0 and closed[t - 1] == 1:
            return t
    return None


def _match_joint(joint_traces: dict[str, np.ndarray], obj_text: str, cursor: int) -> str | None:
    """Pick the articulated joint driven by primitive ``obj_text``.

    Prefer joints whose name shares a keyword with the object/support text; among
    candidates choose the one with the largest qpos range over [cursor, T]. Falls
    back to the globally most-active joint if no keyword match.
    """
    if not joint_traces:
        return None
    text = (obj_text or "").lower()
    kw_hits = [k for k in _JOINT_KEYWORDS if k in text]

    def rng(jn):
        seg = joint_traces[jn][cursor:]
        return float(np.ptp(seg)) if seg.size else 0.0

    matched = [jn for jn in joint_traces if any(k in jn.lower() for k in kw_hits)]
    pool = matched if matched else list(joint_traces)
    pool = [jn for jn in pool if rng(jn) > 1e-3]
    if not pool:
        return None
    return max(pool, key=rng)


def _joint_plateau_frame(trace: np.ndarray, cursor: int, frac: float = 0.9) -> int | None:
    """Frame where the joint change from ``cursor`` first reaches ``frac`` of its final change."""
    seg = trace[cursor:]
    if seg.size < 2:
        return None
    ch = np.abs(seg - seg[0])
    final = abs(seg[-1] - seg[0])
    if final < 1e-4:
        final = float(ch.max())
    if final < 1e-4:
        return None
    idx = int(np.argmax(ch >= frac * final))
    return cursor + idx


def _settle_frame(pos: np.ndarray, cursor: int, move_thr: float = 0.02, still_thr: float = 0.003) -> int | None:
    """Frame where an object, having moved > move_thr since cursor, comes to rest."""
    seg = pos[cursor:]
    if seg.size < 3:
        return None
    disp = np.linalg.norm(seg - seg[0], axis=1)
    if disp.max() < move_thr:
        return None
    vel = np.linalg.norm(np.diff(seg, axis=0), axis=1)
    for k in range(len(vel) - 1, 0, -1):
        if vel[k] > still_thr:
            return cursor + min(k + 1, len(seg) - 1)
    return None


# --------------------------------------------------------------------------- #
# plan-guided boundary location
# --------------------------------------------------------------------------- #
def plan_guided_boundaries(
    T: int,
    plan_ops: Sequence[str],
    plan_objs: Sequence[str],
    gripper_closed: np.ndarray,
    joint_traces: dict[str, np.ndarray],
    object_traces: dict[str, np.ndarray] | None = None,
) -> tuple[list[int], list[str]]:
    """Locate the M-1 internal boundaries, one per primitive completion, in order.

    Returns (boundaries, signal_sources) where signal_sources[i] labels how the end
    of primitive i was found: 'grasp' | 'release' | 'joint:<name>' | 'settle' |
    'even(fallback)'. Boundaries are strictly increasing in (0, T); on a missing
    signal the remaining primitives are spaced evenly so the M-segment invariant holds.
    """
    M = len(plan_ops)
    object_traces = object_traces or {}
    boundaries: list[int] = []
    sources: list[str] = []
    cursor = 0
    for i in range(M - 1):
        op = plan_ops[i]
        obj = plan_objs[i] if i < len(plan_objs) else ""
        b: int | None = None
        src = ""
        if op in GRASP_OPS:
            b = _next_gripper_transition(gripper_closed, cursor, to_closed=True)
            src = "grasp"
        elif op in RELEASE_OPS:
            b = _next_gripper_transition(gripper_closed, cursor, to_closed=False)
            src = "release"
        elif op in JOINT_OPS:
            jn = _match_joint(joint_traces, obj, cursor)
            if jn is not None:
                b = _joint_plateau_frame(joint_traces[jn], cursor)
                src = f"joint:{jn}"
        elif op in SETTLE_OPS:
            best = None
            for n, pos in object_traces.items():
                f = _settle_frame(pos, cursor)
                if f is not None:
                    best = f if best is None else max(best, f)
            b = best
            src = "settle"
        # enforce strictly increasing and in-range; fall back to even spacing
        if b is None or b <= cursor or b >= T:
            remaining = M - i
            step = max(1, (T - cursor) // remaining)
            b = min(cursor + step, T - (M - 1 - i))
            src = (src + "|even" if src else "even") + "(fallback)"
        boundaries.append(int(b))
        sources.append(src)
        cursor = b
    return boundaries, sources


# --------------------------------------------------------------------------- #
# top-level annotator
# --------------------------------------------------------------------------- #
def _plan_op_obj(plan) -> tuple[list[str], list[str]]:
    """Extract ordered (ops, object-or-support-text) from a Plan or list of primitives/strings."""
    prims = plan.real() if hasattr(plan, "real") else plan
    ops, objs = [], []
    for p in prims:
        if isinstance(p, str):
            ops.append(p); objs.append("")
        else:
            ops.append(p.op)
            objs.append(p.support or p.object or "")
    return ops, objs


def resolve_segment_objects(
    segments: list[Segment], object_traces: dict[str, np.ndarray], ops: Sequence[str],
) -> dict[int, dict[str, Any]]:
    """Disambiguate repeated object tokens (e.g. "both moka pots") to concrete instances.

    For each ``pick`` segment, in demo order, assign the movable instance with the
    largest displacement inside that segment's frame span, refusing an instance
    already assigned to an earlier pick round. Returns {seg_index: {instance, disp}}.
    Place segments inherit the instance resolved by their preceding pick.
    """
    resolved: dict[int, dict[str, Any]] = {}
    used: set[str] = set()
    last_pick_instance = None
    for i, seg in enumerate(segments):
        if seg.op in GRASP_OPS:
            best_name, best_disp = None, -1.0
            for n, pos in object_traces.items():
                if n in used:
                    continue
                span = pos[seg.start:seg.end]
                if span.shape[0] < 2:
                    continue
                disp = float(np.linalg.norm(span - span[0], axis=1).max())
                if disp > best_disp:
                    best_name, best_disp = n, disp
            if best_name is not None:
                used.add(best_name)
                last_pick_instance = best_name
                resolved[i] = {"instance": best_name, "disp": round(best_disp, 4)}
        elif seg.op in RELEASE_OPS and last_pick_instance is not None:
            span = object_traces.get(last_pick_instance)
            disp = None
            if span is not None:
                s = span[seg.start:seg.end]
                if s.shape[0] >= 2:
                    disp = round(float(np.linalg.norm(s - s[0], axis=1).max()), 4)
            resolved[i] = {"instance": last_pick_instance, "disp": disp}
    return resolved


def annotate_demo(
    demo_path: str,
    plan,
    bddl_file: str | None = None,
    demo_key: str = "demo_0",
    window: int = 3,
    suite: str | None = None,
    task: str | None = None,
    env: Any = None,
    build_env: bool = True,
    min_hold: int = 12,
    move_eps: float = 0.01,
) -> dict[str, Any]:
    """Replay a LIBERO hdf5 demo -> plan-guided segment annotation dict.

    ``plan`` is a ``vocab.Plan`` (or list of ops). Boundaries are located per the
    plan-guided rules; the returned dict carries the segment intervals, active op,
    the segment-end window, and the signal source used for each boundary, plus the
    hard-invariant flags. Requires the LIBERO sim (mujoco/EGL) only when the plan
    contains joint/settle ops; pure pick/place demos need no env.
    """
    if not os.path.exists(demo_path):
        raise FileNotFoundError(demo_path)
    import h5py

    ops, objs = _plan_op_obj(plan)
    M = len(ops)

    with h5py.File(demo_path, "r") as f:
        d = f["data"][demo_key]
        states = d["states"][:]
        actions = d["actions"][:]
        T = int(states.shape[0])
        instruction = ""
        if "problem_info" in f["data"].attrs:
            instruction = json.loads(f["data"].attrs["problem_info"]).get(
                "language_instruction", "")

    gclosed = gripper_closed_from_command(actions)

    # ambiguous object token: same non-empty object appears on >1 pick (e.g. two moka pots)
    pick_objs = [o for op, o in zip(ops, objs) if op in GRASP_OPS and o]
    ambiguous = len(pick_objs) != len(set(pick_objs))

    joint_traces: dict[str, np.ndarray] = {}
    object_traces: dict[str, np.ndarray] = {}
    need_replay = any(o in JOINT_OPS for o in ops) or any(o in SETTLE_OPS for o in ops)
    # object traces are needed for settle boundaries, instance disambiguation, and the
    # grasp debounce (a short closed run that moves nothing is approach jitter).
    want_objects = (any(o in SETTLE_OPS for o in ops) or ambiguous
                    or any(o in GRASP_OPS for o in ops))
    own_env = False
    if (need_replay or want_objects) and env is None and build_env and bddl_file:
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero.envs import OffScreenRenderEnv
        env = OffScreenRenderEnv(bddl_file_name=bddl_file, camera_heights=128, camera_widths=128)
        env.reset()
        own_env = True
    if env is not None:
        if need_replay:
            joint_traces = replay_joint_traces(env, states)
        if want_objects:
            movable = [n for n in env.env.obj_of_interest if n in env.env.obj_body_id]
            if movable:
                object_traces = replay_object_traces(env, states, movable)
    if own_env:
        env.close()

    gclosed_raw = gclosed
    gclosed, suppressed = debounce_gripper(
        gclosed, min_hold=min_hold, object_traces=object_traces, move_eps=move_eps)

    if M <= 1:
        boundaries, sources = [], []
    else:
        boundaries, sources = plan_guided_boundaries(
            T, ops, objs, gclosed, joint_traces, object_traces)

    seg_signals = sources + [""]  # last segment ends at T (no boundary signal)
    segs = segments_from_boundaries(T, boundaries, ops=ops, window=window, signals=seg_signals)

    # concrete-instance disambiguation for repeated object tokens
    seg_objs = [{} for _ in segs]
    if ambiguous and object_traces:
        resolved = resolve_segment_objects(segs, object_traces, ops)
        for i, r in resolved.items():
            seg_objs[i] = r

    invariants = {
        "count_eq_M": len(segs) == M,
        "ordered_matches_plan": [s.op for s in segs] == list(ops),
        "no_overlap_full_cover": segment_invariants_ok(segs, T),
    }
    invariants["all_ok"] = bool(
        invariants["count_eq_M"] and invariants["ordered_matches_plan"]
        and invariants["no_overlap_full_cover"]
    )
    return {
        "demo_path": demo_path,
        "demo_key": demo_key,
        "suite": suite,
        "task": task,
        "instruction": instruction,
        "T": T,
        "M": M,
        "plan_ops": list(ops),
        "boundaries": boundaries,
        "segments": [dict(s.to_dict(), **({"object_resolved": seg_objs[i]} if seg_objs[i] else {}))
                     for i, s in enumerate(segs)],
        "window": window,
        "object_disambiguated": bool(ambiguous and object_traces),
        "gripper_debounce": {
            "min_hold": min_hold, "move_eps": move_eps,
            "suppressed_runs": suppressed,
            "raw_events": [int(t) for t in np.nonzero(np.diff(gclosed_raw))[0] + 1],
            "used_events": [int(t) for t in np.nonzero(np.diff(gclosed))[0] + 1],
        },
        "invariants": invariants,
    }


def render_review(
    ann: dict[str, Any],
    demo_path: str,
    out_path: str,
    demo_key: str = "demo_0",
    n_cols: int = 6,
    rgb_key: str = "obs/agentview_rgb",
    flip_vertical: bool = True,
) -> str:
    """Render a human-audit image: keyframe grid + segment-boundary strip + op labels.

    Keyframes are sampled uniformly across the demo (plus every boundary frame). A
    thin timeline strip above the grid colours each segment and draws a vertical
    line at each boundary annotated with the primitive label (SPEC task 3 review).
    """
    import h5py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    with h5py.File(demo_path, "r") as f:
        rgb = f["data"][demo_key][rgb_key][:]
    if flip_vertical:
        rgb = rgb[:, ::-1]
    T = ann["T"]
    segs = ann["segments"]
    boundaries = ann["boundaries"]

    # keyframes: uniform sample + boundary frames + segment midpoints
    kf = set(np.linspace(0, T - 1, n_cols * 2, dtype=int).tolist())
    kf.update(int(b) for b in boundaries)
    kf.update(int((s["start"] + s["end"]) // 2) for s in segs)
    keyframes = sorted(k for k in kf if 0 <= k < T)

    n = len(keyframes)
    n_rows = int(np.ceil(n / n_cols))
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(n_cols * 2.0, n_rows * 2.0 + 1.4))
    gs = fig.add_gridspec(n_rows + 1, n_cols, height_ratios=[0.5] + [1] * n_rows)

    # timeline strip
    ax = fig.add_subplot(gs[0, :])
    for i, s in enumerate(segs):
        ax.add_patch(Rectangle((s["start"], 0), s["end"] - s["start"], 1,
                               color=cmap(i % 10), alpha=0.5))
        ax.text((s["start"] + s["end"]) / 2, 0.5, s["op"], ha="center", va="center",
                fontsize=8, fontweight="bold")
    for b in boundaries:
        ax.axvline(b, color="k", lw=1.5)
        ax.text(b, 1.05, str(b), ha="center", va="bottom", fontsize=6)
    ax.set_xlim(0, T); ax.set_ylim(0, 1); ax.set_yticks([])
    ok = ann["invariants"]["all_ok"]
    ax.set_title(f"{ann.get('task','')}  |  M={ann['M']}  T={T}  invariants_ok={ok}\n{ann.get('instruction','')}",
                 fontsize=9)

    def seg_of(t):
        for i, s in enumerate(segs):
            if s["start"] <= t < s["end"]:
                return i
        return len(segs) - 1

    for j, t in enumerate(keyframes):
        r, c = divmod(j, n_cols)
        axi = fig.add_subplot(gs[r + 1, c])
        axi.imshow(rgb[t])
        si = seg_of(t)
        is_b = t in boundaries
        axi.set_title(f"t={t} {segs[si]['op']}" + ("  |B" if is_b else ""),
                      fontsize=7, color="red" if is_b else "black")
        axi.set_xticks([]); axi.set_yticks([])
        for spine in axi.spines.values():
            spine.set_edgecolor(cmap(si % 10)); spine.set_linewidth(2)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=90, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_annotation(ann: dict[str, Any], out_dir: str) -> str:
    """Write an annotation dict to ``<out_dir>/<suite>/<task>.json`` (or task.json)."""
    sub = Path(out_dir)
    if ann.get("suite"):
        sub = sub / ann["suite"]
    sub.mkdir(parents=True, exist_ok=True)
    name = (ann.get("task") or Path(ann["demo_path"]).stem) + ".json"
    path = sub / name
    with open(path, "w") as f:
        json.dump(ann, f, indent=2)
    return str(path)
