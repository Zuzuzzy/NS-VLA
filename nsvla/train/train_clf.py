"""Stage I-a: supervised training of the monotone-pointer classifier (Eq. 5, App. C.2).

Trains ``encoder.primitive_clf.PrimitiveClassifier`` on the annotated segments
(``train.annotate``) using the frozen mean-pooled VLM features
(``encoder.vlm_features``, cached to disk). Per sampled frame - one per H-chunk
boundary - the target is the active primitive of the segment it falls in; class
weights are inverse primitive frequency, which addresses label imbalance only and
leaves the mechanism untouched. The segment-end window ``w`` is swept on held-out
demonstrations.

Frame accuracy alone would overstate the pointer: a classifier can be accurate
per frame and still trigger its boundaries at the wrong moment. The reported
evaluation therefore also runs a *pointer-replay simulation* - the Eq. 5 {stay, +1}
rule iterated over each demo's feature sequence from m0 = 0 - and reports, per suite:
the fraction of demos reaching the final plan slot, the mean absolute deviation (in
frames) of triggered boundaries from ground truth, and the premature-advance rate.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from nsvla.config import Config
from nsvla.encoder.primitive_clf import PrimitiveClassifier
from nsvla.encoder.vlm_features import load_cached_features
from nsvla.primitives.vocab import default_vocab
from nsvla.utils import paths

FEATURE_ROOT = paths.feature_root()
ANNOTATION_ROOT = paths.annotation_root()
RUN_ROOT = str(Path(paths.run_root()) / "pointer")


# --------------------------------------------------------------------------- #
# dataset assembly (features x annotations)
# --------------------------------------------------------------------------- #
@dataclass
class DemoSample:
    suite: str
    task: str
    psi_bar: np.ndarray          # (n_frames_sampled, d)
    frame_indices: np.ndarray    # (n_frames_sampled,) original frame idx
    frame_ops: list[str]         # active op per sampled frame (segment membership)
    boundaries: list[int]        # ground-truth internal boundaries (original frames)
    plan_ops: list[str]          # ordered plan primitives
    T: int


def _op_at_frame(segments: list[dict], t: int) -> str:
    for s in segments:
        if s["start"] <= t < s["end"]:
            return s["op"]
    return segments[-1]["op"]


def build_dataset(
    feature_root: str = FEATURE_ROOT,
    annotation_root: str = ANNOTATION_ROOT,
    suites: list[str] | None = None,
) -> list[DemoSample]:
    """Join cached features with annotations into per-demo samples."""
    ann_root = Path(annotation_root)
    feat_root = Path(feature_root)
    samples: list[DemoSample] = []
    suite_dirs = [d for d in sorted(ann_root.iterdir()) if d.is_dir() and d.name != "review"]
    for sd in suite_dirs:
        if suites and sd.name not in suites:
            continue
        for aj in sorted(sd.glob("*.json")):
            ann = json.loads(aj.read_text())
            task = ann.get("task") or aj.stem
            fpath = feat_root / sd.name / task / "demo.npz"
            if not fpath.exists():
                # fall back to any npz in the task dir
                cand = list((feat_root / sd.name / task).glob("*.npz")) if (feat_root / sd.name / task).exists() else []
                if not cand:
                    continue
                fpath = cand[0]
            feat = load_cached_features(fpath)
            psi = feat["psi_bar"]
            fidx = feat["frame_indices"]
            segs = ann["segments"]
            frame_ops = [_op_at_frame(segs, int(t)) for t in fidx]
            samples.append(DemoSample(
                suite=sd.name, task=task, psi_bar=psi, frame_indices=fidx,
                frame_ops=frame_ops, boundaries=list(ann["boundaries"]),
                plan_ops=list(ann["plan_ops"]), T=int(ann["T"]),
            ))
    return samples


# --------------------------------------------------------------------------- #
# training
# --------------------------------------------------------------------------- #
def _class_weights(labels: np.ndarray, vocab_size: int) -> np.ndarray:
    counts = np.bincount(labels, minlength=vocab_size).astype(np.float64)
    inv = np.zeros_like(counts)
    nz = counts > 0
    inv[nz] = 1.0 / counts[nz]
    inv[nz] = inv[nz] / inv[nz].mean()   # normalize so mean weight ~1 over present classes
    return inv


def _seg_end_boost(sample: DemoSample, w: int) -> np.ndarray:
    """Per sampled frame: 2.0 if in the last w sampled steps of its segment, else 1.0.

    Emphasizes the segment-end window without discarding the dense mid-segment
    supervision. ``w`` is in sampled-step units.
    """
    ops = sample.frame_ops
    n = len(ops)
    boost = np.ones(n, dtype=np.float32)
    # segment end = last index before op changes (or final frame)
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ops[j + 1] == ops[i]:
            j += 1
        for k in range(max(i, j - w + 1), j + 1):
            boost[k] = 2.0
        i = j + 1
    return boost


def train_clf(
    cfg: Config | None = None,
    feature_root: str = FEATURE_ROOT,
    annotation_root: str = ANNOTATION_ROOT,
    run_dir: str = RUN_ROOT,
    epochs: int = 300,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 40,
    holdout_frac: float = 0.2,
    w_sweep: tuple[int, ...] = (2, 3, 4),
    seed: int = 0,
    device: str | None = None,
) -> dict[str, Any]:
    """Fit g_phi; sweep w on held-out demos; run the pointer-replay eval; save ckpt+metrics."""
    import torch
    import torch.nn.functional as F

    cfg = cfg or Config()
    vocab = default_vocab()
    V = len(vocab)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(seed)

    samples = build_dataset(feature_root, annotation_root)
    if not samples:
        raise RuntimeError(f"no (feature, annotation) pairs found under {feature_root} / {annotation_root}")
    d_in = samples[0].psi_bar.shape[1]

    # demo-level held-out split, stratified by suite
    by_suite: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        by_suite.setdefault(s.suite, []).append(i)
    val_idx: set[int] = set()
    for suite, idxs in by_suite.items():
        idxs = list(idxs)
        rng.shuffle(idxs)
        k = max(1, int(round(holdout_frac * len(idxs))))
        val_idx.update(idxs[:k])
    train_samples = [s for i, s in enumerate(samples) if i not in val_idx]
    val_samples = [s for i, s in enumerate(samples) if i in val_idx]

    def stack(sset):
        X = np.concatenate([s.psi_bar for s in sset], axis=0)
        y = np.array([vocab.id(o) for s in sset for o in s.frame_ops], dtype=np.int64)
        return X, y

    Xtr, ytr = stack(train_samples)
    Xva, yva = stack(val_samples)
    cw = _class_weights(ytr, V)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xva_t = torch.tensor(Xva, dtype=torch.float32, device=device)
    yva_t = torch.tensor(yva, dtype=torch.long, device=device)
    cw_t = torch.tensor(cw, dtype=torch.float32, device=device)

    best = {"w": None, "val_acc": -1.0, "state": None}
    sweep_log = {}
    for w in w_sweep:
        boost = np.concatenate([_seg_end_boost(s, w) for s in train_samples], axis=0)
        boost_t = torch.tensor(boost, dtype=torch.float32, device=device)

        torch.manual_seed(seed)
        clf = PrimitiveClassifier(d_in=d_in, vocab_size=V, hidden=cfg.classifier.hidden).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=weight_decay)

        best_va, best_state, bad = -1.0, None, 0
        for ep in range(epochs):
            clf.train()
            logits = clf.logits(Xtr_t)
            per = F.cross_entropy(logits, ytr_t, weight=cw_t, reduction="none")
            loss = (per * boost_t).mean()
            opt.zero_grad(); loss.backward(); opt.step()

            clf.eval()
            with torch.no_grad():
                va_pred = clf.logits(Xva_t).argmax(1)
                va_acc = float((va_pred == yva_t).float().mean()) if len(yva_t) else 0.0
            if va_acc > best_va:
                best_va, bad = va_acc, 0
                best_state = {k: v.detach().clone() for k, v in clf.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        sweep_log[w] = {"val_acc": best_va, "epochs_ran": ep + 1}
        if best_va > best["val_acc"]:
            best = {"w": w, "val_acc": best_va, "state": best_state}

    # rebuild best classifier
    clf = PrimitiveClassifier(d_in=d_in, vocab_size=V, hidden=cfg.classifier.hidden).to(device)
    clf.load_state_dict(best["state"])
    clf.eval()

    # frame-level accuracy (held-out) + overall train
    with torch.no_grad():
        va_acc = float((clf.logits(Xva_t).argmax(1) == yva_t).float().mean()) if len(yva_t) else 0.0
        tr_acc = float((clf.logits(Xtr_t).argmax(1) == ytr_t).float().mean())

    # pointer-replay evaluation on ALL demos, per suite
    replay = pointer_replay_eval(clf, samples, vocab, device, tol_frames=8)

    # save
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": clf.state_dict(), "d_in": d_in, "vocab_size": V,
        "hidden": cfg.classifier.hidden, "best_w": best["w"], "ops": vocab.ops,
    }, out / "clf.pt")
    metrics = {
        "n_demos": len(samples),
        "n_train_frames": int(len(ytr)), "n_val_frames": int(len(yva)),
        "n_val_demos": len(val_samples),
        "d_in": d_in, "vocab_size": V,
        "best_w": best["w"], "w_sweep": sweep_log,
        "frame_acc_heldout": va_acc, "frame_acc_train": tr_acc,
        "class_weights": {vocab.op(i): float(cw[i]) for i in range(V)},
        "pointer_replay": replay,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    cfg.classifier.seg_window = int(best["w"])
    return metrics


# --------------------------------------------------------------------------- #
# pointer-replay simulation (Eq. 5) — honest boundary-timing eval
# --------------------------------------------------------------------------- #
def pointer_replay_eval(clf, samples, vocab, device, tol_frames: int = 8) -> dict[str, Any]:
    """Iterate the {stay,+1} monotone pointer over each demo and score against truth.

    Per suite (and overall) reports:
      * reach_final_rate : fraction of demos whose pointer reaches the last plan slot.
      * mean_boundary_dev_frames : mean |predicted advance frame - true boundary| over
        matched (k-th) boundaries, in original frame units.
      * premature_rate : fraction of triggered advances firing > tol_frames before truth.
      * frame_op_acc : per-frame active-op argmax accuracy.
    """
    import torch

    per_suite: dict[str, dict[str, list]] = {}
    for s in samples:
        M = len(s.plan_ops)
        plan_ids = [vocab.id(o) for o in s.plan_ops]
        plan_ids_t = torch.tensor([plan_ids], dtype=torch.long, device=device)
        plan_len_t = torch.tensor([M], dtype=torch.long, device=device)
        m_prev = torch.tensor([0], dtype=torch.long, device=device)

        adv_frames: list[int] = []
        op_correct = 0
        with torch.no_grad():
            psi = torch.tensor(s.psi_bar, dtype=torch.float32, device=device)
            vocab_logits = clf.logits(psi)          # (n, V) for frame-op acc
            frame_pred = vocab_logits.argmax(1).cpu().numpy()
            for t in range(psi.shape[0]):
                m_t, b_t, _ = clf.step(psi[t:t+1], plan_ids_t, m_prev, plan_len_t)
                if int(b_t.item()) == 1:
                    adv_frames.append(int(s.frame_indices[t]))
                m_prev = m_t
            final_m = int(m_prev.item())

        # frame-op accuracy vs segment-membership label
        true_ids = np.array([vocab.id(o) for o in s.frame_ops])
        op_correct = int((frame_pred == true_ids).sum())

        # match k-th advance to k-th true boundary
        devs = []
        premature = 0
        for k, tb in enumerate(s.boundaries):
            if k < len(adv_frames):
                dev = adv_frames[k] - tb
                devs.append(abs(dev))
                if dev < -tol_frames:
                    premature += 1

        d = per_suite.setdefault(s.suite, {
            "reach": [], "dev": [], "premature": [], "n_adv": [], "n_true_b": [],
            "op_correct": [], "op_total": []})
        d["reach"].append(1.0 if final_m == M - 1 else 0.0)
        d["dev"].extend(devs)
        d["premature"].append(premature)
        d["n_adv"].append(len(adv_frames))
        d["n_true_b"].append(len(s.boundaries))
        d["op_correct"].append(op_correct)
        d["op_total"].append(len(s.frame_ops))

    def summarize(d):
        n_true_b = sum(d["n_true_b"])
        return {
            "n_demos": len(d["reach"]),
            "reach_final_rate": float(np.mean(d["reach"])) if d["reach"] else 0.0,
            "mean_boundary_dev_frames": float(np.mean(d["dev"])) if d["dev"] else 0.0,
            "median_boundary_dev_frames": float(np.median(d["dev"])) if d["dev"] else 0.0,
            "premature_rate": float(sum(d["premature"]) / n_true_b) if n_true_b else 0.0,
            "avg_advances": float(np.mean(d["n_adv"])) if d["n_adv"] else 0.0,
            "avg_true_boundaries": float(np.mean(d["n_true_b"])) if d["n_true_b"] else 0.0,
            "frame_op_acc": float(sum(d["op_correct"]) / sum(d["op_total"])) if sum(d["op_total"]) else 0.0,
        }

    result = {suite: summarize(d) for suite, d in sorted(per_suite.items())}
    # overall
    alld = {"reach": [], "dev": [], "premature": [], "n_adv": [], "n_true_b": [], "op_correct": [], "op_total": []}
    for d in per_suite.values():
        for k in alld:
            alld[k].extend(d[k])
    result["overall"] = summarize(alld)
    return result


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Stage I-a pointer-classifier training")
    p.add_argument("--feature-root", default=FEATURE_ROOT)
    p.add_argument("--annotation-root", default=ANNOTATION_ROOT)
    p.add_argument("--run-dir", default=RUN_ROOT)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    m = train_clf(feature_root=args.feature_root, annotation_root=args.annotation_root,
                  run_dir=args.run_dir, seed=args.seed)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
