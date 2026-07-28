#!/usr/bin/env python
"""Cache frozen-VLM context features for the annotated demonstrations.

    python data/extract_features.py --manifest data/manifests/libero_1shot.json \
        --out data/features

The encoder is frozen, so its features are extracted once and reused by every Stage-I
run. One feature vector is taken per H-chunk boundary, which is exactly the rate at
which the pointer makes a decision at rollout time; sampling denser would train the
pointer on frames it will never be asked about.

Images are read from the demonstration file itself and vertically flipped to match the
orientation the live simulator returns, so training and rollout features agree.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    from nsvla.utils import paths

    p = argparse.ArgumentParser(description="Frozen-VLM feature extraction")
    p.add_argument("--manifest", default="data/manifests/libero_1shot.json")
    p.add_argument("--out", default=paths.feature_root())
    p.add_argument("--model", default=paths.vlm_model_dir())
    p.add_argument("--chunk-h", type=int, default=8, help="sample one frame per H steps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--rgb-key", default="obs/agentview_rgb")
    p.add_argument("--save-visual-tokens", action="store_true",
                   help="also cache visual tokens for the App. F sparsifier")
    p.add_argument("--topk-visual", type=int, default=32)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--suite", default=None)
    p.add_argument("--device", default="cuda")
    return p


def main() -> None:
    args = build_parser().parse_args()

    import h5py

    from nsvla.encoder.vlm_features import VLMFeatureEncoder

    manifest = json.loads(Path(args.manifest).read_text())
    entries = [e for e in manifest["entries"] if e.get("exists", True)]
    if args.suite:
        entries = [e for e in entries if e["suite"] == args.suite]
    if args.limit:
        entries = entries[: args.limit]

    encoder = VLMFeatureEncoder(model_id=args.model, device=args.device)
    encoder.load()

    for e in entries:
        with h5py.File(e["demo_path"], "r") as f:
            rgb = f["data"][e["demo_key"]][args.rgb_key][:]
        rgb = rgb[:, ::-1]      # match the live simulator's vertical convention
        out_path = Path(args.out) / e["suite"] / e["task"] / "demo.npz"
        info = encoder.cache_demo(
            list(rgb), e["instruction"], out_path,
            chunk_H=args.chunk_h, batch_size=args.batch_size,
            save_visual_tokens=args.save_visual_tokens,
            topk_visual=args.topk_visual if args.save_visual_tokens else None,
        )
        print(f"[features] {e['suite']}/{e['task']} frames={info['n_sampled']} "
              f"d={info['d']} -> {info['out_path']}", flush=True)

    print(f"[features] {len(entries)} demonstrations cached under {args.out}")


if __name__ == "__main__":
    main()
