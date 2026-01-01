#!/usr/bin/env python
"""Serve an action solver over ZMQ.

    python scripts/solver_server.py --solver vla_adapter \
        --checkpoint checkpoints/vla-adapter/spatial --port 5678

Run this inside whatever environment the solver's weights need; the encoder side
reaches it through ``nsvla.solver.remote.RemoteSolver`` and needs none of those
dependencies. ``--trainable`` additionally attaches the LoRA + action-head optimizer
that Stage-II AWR updates, and is required by ``scripts/train_rl.py``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NS-VLA action-solver server")
    p.add_argument("--solver", default="vla_adapter",
                   help="solver name: vla_adapter (reference), fake (CPU stand-in), "
                        "or any solver you registered with @register_solver")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5678)
    p.add_argument("--checkpoint", default=None, help="solver weights")
    p.add_argument("--source-root", default=None, help="solver source tree")
    p.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    p.add_argument("--task-suite", default="libero_spatial")
    p.add_argument("--use-pro-version", action="store_true")
    p.add_argument("--chunk-h", type=int, default=8, help="action chunk length H")
    p.add_argument("--action-dim", type=int, default=7)
    p.add_argument("--quiet", action="store_true", help="do not log every request")
    # Stage-II online RL
    p.add_argument("--trainable", action="store_true",
                   help="attach the low-level optimizer (required by scripts/train_rl.py)")
    p.add_argument("--freeze-theta", action="store_true",
                   help="ablation: keep the sampling wire but never update theta")
    p.add_argument("--lora-rank", type=int, default=64)
    p.add_argument("--lora-target", choices=["all-linear", "llm"], default="all-linear")
    p.add_argument("--low-lr", type=float, default=1e-5)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--sample-cache", type=int, default=4096)
    p.add_argument("--sample-seed", type=int, default=0)
    p.add_argument("--max-vram-mb", type=int, default=0,
                   help="cap this process's CUDA allocator when the card is shared")
    return p


def main() -> None:
    args = build_parser().parse_args()

    from nsvla.config import SolverConfig
    from nsvla.solver import make_solver
    from nsvla.solver.server import serve

    cfg = SolverConfig(
        name=args.solver,
        checkpoint=args.checkpoint or "",
        unnorm_key=args.unnorm_key,
        task_suite=args.task_suite,
        use_pro_version=args.use_pro_version,
        source_root=args.source_root or "",
        chunk_H=args.chunk_h,
        action_dim=args.action_dim,
    )
    extra = {}
    if args.solver == "vla_adapter":
        extra = dict(
            trainable=args.trainable, freeze_theta=args.freeze_theta,
            lora_rank=args.lora_rank, lora_target=args.lora_target,
            low_lr=args.low_lr, grad_clip=args.grad_clip,
            sample_cache=args.sample_cache, sample_seed=args.sample_seed,
            max_vram_mb=args.max_vram_mb,
        )
    serve(make_solver(cfg, **extra), host=args.host, port=args.port, verbose=not args.quiet)


if __name__ == "__main__":
    main()
