#!/usr/bin/env python
"""Evaluate a policy on LIBERO / LIBERO-Plus and write traces + metrics.

    python scripts/run_eval.py --config configs/eval/libero.yaml \
        --checkpoint runs/pointer/clf.pt

Traces are the only metric source: this writes one JSON trace per episode and then
computes the metrics from those traces, so every number can be recomputed offline.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NS-VLA evaluation")
    p.add_argument("--config", required=True, help="evaluation config YAML")
    p.add_argument("--checkpoint", default=None,
                   help="pointer checkpoint (clf.pt or pointer_config.json)")
    p.add_argument("--exp", default=None, help="override the run name")
    p.add_argument("--policy", default=None, choices=["nsvla", "solver_bare"])
    p.add_argument("--suite", default=None, help="evaluate a single suite")
    p.add_argument("--tasks", type=int, default=None, help="cap the number of tasks")
    p.add_argument("--task-ids", default=None, help="comma-separated explicit task ids")
    p.add_argument("--episodes", type=int, default=None, help="episodes per task")
    p.add_argument("--solver", default=None, choices=["remote", "vla_adapter", "fake"])
    p.add_argument("--solver-port", type=int, default=None)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from nsvla.config import Config
    from nsvla.eval.metrics import summarize
    from nsvla.eval.run_suite import DEFAULT_CLF_PATH, run_suite

    cfg = Config.from_yaml(args.config)
    if args.exp:
        cfg.exp_name = args.exp
    if args.policy:
        cfg.eval.policy = args.policy
    if args.suite:
        cfg.env.suites = [args.suite]
    if args.episodes is not None:
        cfg.eval.episodes_per_task = args.episodes
    if args.solver:
        cfg.solver.name = args.solver
    if args.solver_port is not None:
        cfg.solver.port = args.solver_port

    # The fake solver exists to exercise the wiring, and there is no trained pointer to
    # go with it. Without a checkpoint the nsvla policy could only fail, so fall back to
    # the pointer-free policy: that keeps the documented smoke command self-contained
    # and CPU-only (no pointer checkpoint, no frozen encoder, no solver weights).
    if cfg.solver.name == "fake" and args.checkpoint is None and cfg.eval.policy == "nsvla":
        cfg.eval.policy = "solver_bare"
        print("[eval] --solver fake without --checkpoint: evaluating the wiring with "
              "policy 'solver_bare' (no pointer, no encoder). Pass --checkpoint to run "
              "the full nsvla policy.", flush=True)

    task_ids = [int(x) for x in args.task_ids.split(",")] if args.task_ids else None
    traces = run_suite(
        cfg,
        policy=cfg.eval.policy,
        n_tasks=args.tasks,
        n_episodes=args.episodes,
        task_ids=task_ids,
        clf_path=args.checkpoint or DEFAULT_CLF_PATH,
        device=args.device,
    )

    metrics = summarize(traces)
    out = cfg.run_dir / "metrics.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))
    print(f"[eval] {len(traces)} traces -> {cfg.run_dir / 'traces'}", flush=True)


if __name__ == "__main__":
    main()
