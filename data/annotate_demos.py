#!/usr/bin/env python
"""Annotate the 1-shot demonstrations into ordered primitive segments.

    python data/annotate_demos.py --manifest data/manifests/libero_1shot.json \
        --out data/primitive_annotations

For every demonstration: extract the symbolic plan from the instruction (Eq. 4), then
locate the M-1 internal segment boundaries from gripper events, articulated-joint
plateaus and object-motion settle. The plan fixes the number and order of segments, so
"segments match the plan, cover every frame, and never overlap" holds by construction;
the per-demo ``invariants.all_ok`` flag records that it did.

Segment boundaries are the supervision for the monotone pointer, so a demonstration
whose invariants fail must be inspected rather than silently trained on: the run
prints a summary and exits non-zero if any demonstration failed.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def build_parser() -> argparse.ArgumentParser:
    from nsvla.utils import paths

    p = argparse.ArgumentParser(description="Plan-guided demonstration annotation")
    p.add_argument("--manifest", default="data/manifests/libero_1shot.json")
    p.add_argument("--out", default=paths.annotation_root())
    p.add_argument("--window", type=int, default=3, help="segment-end label window w")
    p.add_argument("--limit", type=int, default=None, help="annotate only the first N demos")
    p.add_argument("--suite", default=None, help="restrict to one suite")
    p.add_argument("--review-dir", default=None,
                   help="also render a keyframe audit image per demonstration")
    return p


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    from nsvla.encoder.plan_extractor import extract_plan
    from nsvla.train.annotate import annotate_demo, render_review, save_annotation

    manifest = json.loads(Path(args.manifest).read_text())
    entries = [e for e in manifest["entries"] if e.get("exists", True)]
    if args.suite:
        entries = [e for e in entries if e["suite"] == args.suite]
    if args.limit:
        entries = entries[: args.limit]

    failed = []
    for e in entries:
        plan = extract_plan(e["instruction"])
        ann = annotate_demo(
            e["demo_path"], plan,
            bddl_file=e["bddl_file"], demo_key=e["demo_key"],
            window=args.window, suite=e["suite"], task=e["task"],
        )
        path = save_annotation(ann, args.out)
        ok = ann["invariants"]["all_ok"]
        print(f"[annotate] {e['suite']}/{e['task']} M={ann['M']} T={ann['T']} "
              f"boundaries={ann['boundaries']} ok={ok} -> {path}", flush=True)
        if not ok:
            failed.append(f"{e['suite']}/{e['task']}")
        if args.review_dir:
            out_png = Path(args.review_dir) / e["suite"] / f"{e['task']}.png"
            render_review(ann, e["demo_path"], str(out_png), demo_key=e["demo_key"])

    print(f"[annotate] {len(entries) - len(failed)}/{len(entries)} demonstrations passed "
          f"the segment invariants")
    if failed:
        print("[annotate] FAILED: " + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
