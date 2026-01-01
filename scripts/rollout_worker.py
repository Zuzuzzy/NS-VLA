#!/usr/bin/env python
"""Persistent Stage-II rollout worker. Spawned by ``nsvla.rl.train_rl``, not by hand.

Loading the frozen VLM and building a simulator environment costs one to two minutes,
while an Alg. 1 iteration is only a handful of episodes, so a spawn-per-iteration
design would spend most of its wall clock loading. The worker boots once and then
consumes job files until it sees a ``stop`` marker.

Protocol - plain files, so a run is restartable and inspectable with no extra daemon::

    in   <jobs>/w<id>/job_<n>.json   {iteration, mode, suite, task_id, sigma, out_dir,
                                      episodes: [{rollout_id, init_state_idx, seed}]}
    out  <jobs>/w<id>/job_<n>.done   {results: [...], wall_s}
         <jobs>/w<id>/job_<n>.err    traceback on failure; the trainer re-raises it
         <out_dir>/<rollout_id>.json + .npz   trace + update tensors

Thread caps are set BEFORE torch and numpy are imported, and the trainer additionally
launches the worker under ``nice``: one worker holds one frozen VLM and one simulator,
and a fleet of unthrottled workers will thrash the machine long before it runs out of
memory.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path


def _cap_threads(n: int) -> None:
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="NS-VLA Stage-II rollout worker")
    p.add_argument("--worker-id", type=int, required=True)
    p.add_argument("--jobs-dir", required=True)
    p.add_argument("--port", type=int, default=5678)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--clf-path", required=True)
    p.add_argument("--suite", default="libero_spatial")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--poll-s", type=float, default=1.0)
    p.add_argument("--idle-timeout-s", type=float, default=7200.0)
    return p


def main() -> None:
    args = build_parser().parse_args()

    _cap_threads(args.threads)
    os.environ.setdefault("MUJOCO_GL", "egl")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    torch.set_num_threads(args.threads)

    from nsvla.config import Config
    from nsvla.eval.run_suite import build_components
    from nsvla.rl.rollout import rl_rollout_episode, save_rollout

    wdir = Path(args.jobs_dir) / f"w{args.worker_id}"
    wdir.mkdir(parents=True, exist_ok=True)

    cfg = Config()
    cfg.env.suites = [args.suite]
    cfg.solver.port = args.port
    cfg.solver.host = args.host

    t0 = time.time()
    comp = build_components(cfg, "nsvla", clf_path=args.clf_path)
    ping = comp.solver.ping()
    print(f"[w{args.worker_id}] ready in {time.time() - t0:.1f}s device={comp.device} "
          f"ping={ping}", flush=True)

    from libero.libero import benchmark

    from nsvla.envs.libero_env import LiberoEnv

    suite_obj = benchmark.get_benchmark_dict()[args.suite]()
    env = LiberoEnv(cfg.env, image_size=cfg.control.image_size)
    max_steps = env.max_steps(args.suite)

    (wdir / "ready").write_text(json.dumps({"pid": os.getpid(), "ping": ping}))

    last_work = time.time()
    try:
        while True:
            if (wdir / "stop").exists():
                print(f"[w{args.worker_id}] stop flag; exiting", flush=True)
                break
            jobs = sorted(p for p in wdir.glob("job_*.json")
                          if not (p.parent / (p.stem + ".done")).exists()
                          and not (p.parent / (p.stem + ".err")).exists())
            if not jobs:
                if time.time() - last_work > args.idle_timeout_s:
                    print(f"[w{args.worker_id}] idle timeout; exiting", flush=True)
                    break
                time.sleep(args.poll_s)
                continue
            job_path = jobs[0]
            last_work = time.time()
            try:
                _run_job(cfg, job_path, suite_obj, env, comp, max_steps,
                         rl_rollout_episode, save_rollout, args.worker_id)
            except Exception:
                (job_path.parent / (job_path.stem + ".err")).write_text(traceback.format_exc())
                print(f"[w{args.worker_id}] JOB FAILED {job_path.name}\n"
                      f"{traceback.format_exc()}", flush=True)
    finally:
        env.close()


def _run_job(cfg, job_path, suite_obj, env, comp, max_steps,
             rl_rollout_episode, save_rollout, wid) -> None:
    import torch

    job = json.loads(job_path.read_text())
    cfg.rl.rollout.high_sample = bool(job.get("high_sample", cfg.rl.rollout.high_sample))

    # phi lives in the trainer; pick up its latest version. The check is version-gated,
    # so a job that did not move phi costs nothing. theta lives in the solver process
    # and is therefore always current without any sync here.
    pc, pv = job.get("pointer_ckpt"), int(job.get("pointer_version", 0))
    if pc and Path(pc).exists() and pv != getattr(comp, "_pointer_version", -1):
        sd = torch.load(pc, map_location=comp.device)
        comp.clf.load_state_dict(sd.get("state_dict", sd))
        comp.clf.eval()
        comp._pointer_version = pv
        print(f"[w{wid}] loaded pointer version {pv}", flush=True)

    task_id = int(job["task_id"])
    task = suite_obj.get_task(task_id)
    init_states = suite_obj.get_task_init_states(task_id)
    out_dir = Path(job["out_dir"])
    sigma = float(job["sigma"])
    results = []
    t0 = time.time()
    for ep in job["episodes"]:
        trace, feats = rl_rollout_episode(
            cfg, env, task, job["suite"], comp,
            init_state=init_states[int(ep["init_state_idx"]) % len(init_states)],
            seed=int(ep["seed"]),
            rollout_id=str(ep["rollout_id"]),
            sigma=sigma,
            max_steps=max_steps,
            iteration=int(job.get("iteration", 0)),
            group_index=int(ep.get("group_index", 0)),
            mode=job.get("mode", "train"),
            verbose=True,
        )
        paths = save_rollout(trace, feats, out_dir, str(ep["rollout_id"]))
        results.append({
            "rollout_id": ep["rollout_id"],
            "success": bool(trace.success),
            "episode_len": trace.episode_len,
            "n_decisions": len(trace.steps),
            **paths,
        })
    (job_path.parent / (job_path.stem + ".done")).write_text(json.dumps({
        "worker": wid, "results": results, "wall_s": time.time() - t0,
    }, indent=2))
    print(f"[w{wid}] job {job_path.name} done: {len(results)} episodes in "
          f"{time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
