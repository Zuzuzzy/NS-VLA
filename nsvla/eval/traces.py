"""Episode trace schema - the single source of all metrics.

Every episode (eval AND RL rollout) writes one JSON trace; metrics are computed
offline from traces only (rollout code never computes a metric). Schema:

    {task_id, suite, seed, perturb_axis, perturb_level, instruction,
     plan: [{op, args}],
     steps: [{t, m_t, b_t, sub_instr, action_chunk_hash,
              r_task, r_seg, phi, r_low_terms}],
     success, episode_len, wall_time}

Dataclasses with a JSON round-trip; action chunks are hashed rather than stored, to
keep traces small and comparable.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1


def hash_action_chunk(chunk: Any) -> str:
    """Stable content hash of an action chunk (H x action_dim), for the trace."""
    arr = np.ascontiguousarray(np.asarray(chunk, dtype=np.float32))
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


@dataclass
class Step:
    """One decision step (chunk boundary) of an episode."""

    t: int
    m_t: int                       # plan pointer after this step
    b_t: int                       # segment-boundary flag 1[m_t != m_{t-1}]
    action_chunk_hash: str
    sub_instr: str | None = None
    r_task: float = 0.0            # sparse terminal success reward at this step
    r_seg: float = 0.0             # segment/milestone reward (lambda_seg * b_t)
    phi: float = 0.0              # prototype shaping potential Phi_t
    r_low_terms: dict[str, float] = field(default_factory=dict)  # prog/sub/sm/gr breakdown
    # Physical env probe at this decision step (grasp/lift state of the pick target).
    # Diagnostic only — never an input to the policy. Used offline to test whether a
    # pointer advance happened BEFORE the target was actually grasped ("premature
    # advance"). Keys: grip_gap, obj_lift, grasped, eef_z. See run_suite._probe_env.
    probe: dict[str, float] = field(default_factory=dict)

    # ---- Stage-II RL extension (Alg.1). All default to the eval-time neutral value,
    # so an eval trace and an RL trace share one schema and one reader.
    logp: float = 0.0          # log pi^h_phi(u_t | H_t) of the executed slot (Eq. 9 high)
    r_low: float = 0.0         # assembled dense chunk reward r^l_t (Eq. 8 low)
    adv_low: float = 0.0       # chunk-level advantage A^l_t feeding the AWR weight
    sample_id: str | None = None   # solver-side handle of this chunk's cached observation

    def to_dict(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "m_t": self.m_t,
            "b_t": self.b_t,
            "action_chunk_hash": self.action_chunk_hash,
            "sub_instr": self.sub_instr,
            "r_task": self.r_task,
            "r_seg": self.r_seg,
            "phi": self.phi,
            "r_low_terms": self.r_low_terms,
            "probe": self.probe,
            "logp": self.logp,
            "r_low": self.r_low,
            "adv_low": self.adv_low,
            "sample_id": self.sample_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Step":
        return cls(
            t=d["t"],
            m_t=d["m_t"],
            b_t=d["b_t"],
            action_chunk_hash=d["action_chunk_hash"],
            sub_instr=d.get("sub_instr"),
            r_task=d.get("r_task", 0.0),
            r_seg=d.get("r_seg", 0.0),
            phi=d.get("phi", 0.0),
            r_low_terms=d.get("r_low_terms", {}),
            probe=d.get("probe", {}),
            logp=d.get("logp", 0.0),
            r_low=d.get("r_low", 0.0),
            adv_low=d.get("adv_low", 0.0),
            sample_id=d.get("sample_id"),
        )


@dataclass
class EpisodeTrace:
    """One episode's full symbolic trace. Shared by eval and RL rollout."""

    task_id: str
    suite: str
    # NOT the run seed. This is the **episode / init-state index** within the task
    # (0..episodes_per_task-1). The RUN seed lives only in the run directory name's
    # ``_s<k>`` suffix; parse it with ``nsvla.utils.runs.seed_of``. Grouping episodes
    # by this field would yield one cell per init state, not one per run seed.
    seed: int
    instruction: str
    plan: list[dict[str, Any]] = field(default_factory=list)   # [{op, args}]
    steps: list[Step] = field(default_factory=list)
    success: bool = False
    episode_len: int = 0
    wall_time: float = 0.0
    perturb_axis: str | None = None
    perturb_level: int | None = None
    schema_version: int = SCHEMA_VERSION

    # ---- Stage-II RL extension (Alg.1). ``rl`` is null for eval traces.
    # {iteration, group_index, rollout_id, sigma, R_high, adv_high, w_high,
    #  return_low, feature_path, mode}
    rl: dict[str, Any] | None = None

    def add_step(self, step: Step) -> None:
        self.steps.append(step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "suite": self.suite,
            "seed": self.seed,
            "perturb_axis": self.perturb_axis,
            "perturb_level": self.perturb_level,
            "instruction": self.instruction,
            "plan": self.plan,
            "steps": [s.to_dict() for s in self.steps],
            "success": self.success,
            "episode_len": self.episode_len,
            "wall_time": self.wall_time,
            "rl": self.rl,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "EpisodeTrace":
        return cls(
            task_id=d["task_id"],
            suite=d["suite"],
            seed=d["seed"],
            instruction=d["instruction"],
            plan=d.get("plan", []),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            success=d.get("success", False),
            episode_len=d.get("episode_len", 0),
            wall_time=d.get("wall_time", 0.0),
            perturb_axis=d.get("perturb_axis"),
            perturb_level=d.get("perturb_level"),
            schema_version=d.get("schema_version", SCHEMA_VERSION),
            rl=d.get("rl"),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "EpisodeTrace":
        with Path(path).open() as f:
            return cls.from_dict(json.load(f))


def load_traces(directory: str | Path) -> list[EpisodeTrace]:
    """Load every ``*.json`` trace in a directory (metrics input)."""
    return [EpisodeTrace.load(p) for p in sorted(Path(directory).glob("*.json"))]
