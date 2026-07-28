#!/usr/bin/env python
"""Stage I: supervised pretraining.

    python scripts/train_supervised.py --config configs/train/pretrain.yaml

Two sub-stages, selected with ``--stage``:

  ``pointer`` (default) fits the monotone-pointer classifier g_phi (Eq. 5) on the
  annotated primitive segments and the cached frozen-VLM features, then reports both
  held-out frame accuracy and a pointer-replay simulation of boundary timing.

  ``bc`` runs the 1-shot behaviour-cloning warm-start of the low-level policy and
  freezes the result as the KL anchor used by Stage II. It updates the solver's own
  parameters, so it runs inside the solver's environment.

Inputs come from ``scripts/annotate_demos.py`` (segments) and the feature cache
written by ``nsvla.encoder.vlm_features``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NS-VLA Stage-I supervised training")
    p.add_argument("--config", required=True, help="training config YAML")
    p.add_argument("--stage", default="pointer", choices=["pointer", "bc"])
    p.add_argument("--feature-root", default=None, help="cached VLM features")
    p.add_argument("--annotation-root", default=None, help="primitive segment annotations")
    p.add_argument("--out", default=None, help="output directory (default: <run_root>/<exp>)")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--max-steps", type=int, default=None,
                   help="alias for --epochs, for a short smoke run")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()

    from nsvla.config import Config
    from nsvla.train.train_clf import ANNOTATION_ROOT, FEATURE_ROOT, train_clf

    cfg = Config.from_yaml(args.config)
    if args.seed is not None:
        cfg.seed = args.seed
    out = args.out or str(Path(cfg.run_root) / cfg.exp_name)

    if args.stage == "bc":
        from nsvla.train.train_bc import train_bc

        train_bc(cfg)
        return

    metrics = train_clf(
        cfg,
        feature_root=args.feature_root or FEATURE_ROOT,
        annotation_root=args.annotation_root or ANNOTATION_ROOT,
        run_dir=out,
        epochs=args.max_steps or args.epochs,
        seed=cfg.seed,
        device=args.device,
    )
    print(json.dumps(metrics, indent=2))
    print(f"[stage-I] pointer checkpoint -> {Path(out) / 'clf.pt'}", flush=True)


if __name__ == "__main__":
    main()
