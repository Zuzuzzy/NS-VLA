#!/usr/bin/env python
"""Stage II: online RL (H-GRPO on the pointer + AWR on the solver, Alg. 1).

    python scripts/train_rl.py --config configs/train/rl_grpo.yaml \
        --checkpoint runs/pointer/clf.pt

Requires a *trainable* action solver already listening (see
``scripts/solver_server.py --trainable``); the trainer refuses to start otherwise,
because a frozen solver would silently turn Eq. 9 into a high-level-only update.
Rollout workers are spawned by the trainer itself.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NS-VLA Stage-II online RL")
    p.add_argument("--config", required=True, help="RL config YAML")
    p.add_argument("--checkpoint", default=None,
                   help="Stage-I pointer checkpoint; also the frozen BC anchor")
    p.add_argument("--exp", default=None, help="override the run name")
    p.add_argument("--suite", default=None)
    p.add_argument("--task-ids", default=None, help="comma-separated; default = all tasks")
    p.add_argument("--solver-port", type=int, default=None)
    p.add_argument("--worker-gpus", default="0", help="comma-separated GPU ids for the workers")
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None, help="alias for --iters")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from nsvla.config import Config
    from nsvla.rl.train_rl import DEFAULT_POINTER, train_rl

    cfg = Config.from_yaml(args.config)
    if args.exp:
        cfg.exp_name = args.exp
    if args.seed is not None:
        cfg.seed = args.seed
    if args.suite:
        cfg.env.suites = [args.suite]
    if args.workers is not None:
        cfg.rl.rollout.n_workers = args.workers
    iters = args.max_steps or args.iters
    if iters is not None:
        cfg.rl.max_iters = iters
    if args.solver_port is not None:
        cfg.solver.port = args.solver_port

    summary = train_rl(
        cfg,
        clf_path=args.checkpoint or DEFAULT_POINTER,
        task_ids=[int(x) for x in args.task_ids.split(",")] if args.task_ids else None,
        worker_gpus=[g.strip() for g in args.worker_gpus.split(",")],
        port=cfg.solver.port,
        max_iters=cfg.rl.max_iters,
        resume=not args.no_resume,
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "history"}, indent=2))


if __name__ == "__main__":
    main()
