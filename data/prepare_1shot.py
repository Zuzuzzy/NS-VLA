#!/usr/bin/env python
"""Select the 1-shot demonstration for every task and write a manifest.

    python data/prepare_1shot.py --libero-root third_party/LIBERO \
        --out data/manifests/libero_1shot.json

One demonstration per task, chosen deterministically (``--demo-index``, default 0) and
recorded, so the same episode backs the segment annotations, the pointer supervision
and the behaviour-cloning warm-start. The manifest is the single input every later
data step reads; nothing downstream re-selects a demo.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def build_parser() -> argparse.ArgumentParser:
    from nsvla.utils import paths

    p = argparse.ArgumentParser(description="Build the 1-shot demonstration manifest")
    p.add_argument("--libero-root", default=paths.libero_root())
    p.add_argument("--dataset-root", default=None,
                   help="hdf5 root (default: <libero-root>/libero/datasets)")
    p.add_argument("--suites", default=",".join(SUITES))
    p.add_argument("--demo-index", type=int, default=0,
                   help="which demonstration of each task to keep")
    p.add_argument("--out", default="data/manifests/libero_1shot.json")
    return p


def instruction_of(bddl_path: Path) -> str:
    m = re.search(r"\(:language (.+?)\)", bddl_path.read_text())
    return m.group(1).strip() if m else ""


def main() -> None:
    args = build_parser().parse_args()
    root = Path(args.libero_root)
    ds_root = Path(args.dataset_root) if args.dataset_root else root / "libero/datasets"
    bddl_root = root / "libero/libero/bddl_files"

    entries = []
    for suite in [s.strip() for s in args.suites.split(",") if s.strip()]:
        bddls = sorted((bddl_root / suite).glob("*.bddl"))
        if not bddls:
            raise FileNotFoundError(f"no bddl files under {bddl_root / suite}")
        for bddl in bddls:
            demo = ds_root / suite / f"{bddl.stem}_demo.hdf5"
            entries.append({
                "suite": suite,
                "task": bddl.stem,
                "bddl_file": str(bddl),
                "demo_path": str(demo),
                "demo_key": f"demo_{args.demo_index}",
                "instruction": instruction_of(bddl),
                "exists": demo.exists(),
            })

    missing = [e["demo_path"] for e in entries if not e["exists"]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"demo_index": args.demo_index, "entries": entries}, indent=2))
    print(f"[1-shot] {len(entries)} tasks -> {out}")
    if missing:
        print(f"[1-shot] WARNING {len(missing)} demonstration files are missing, "
              f"first: {missing[0]}")


if __name__ == "__main__":
    main()
