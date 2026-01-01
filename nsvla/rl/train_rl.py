"""Stage-II online optimization main loop — NS-VLA §4.3, Appendix C, Algorithm 1.

    iteration -> (every U) prototype refresh -> sample instruction -> collect G rollouts
              -> segment buffers -> group-normalized advantages -> bi-granular update
              -> periodic eval probe -> ckpt / trace / metrics

PROCESS TOPOLOGY

    trainer (this file)
      |- N rollout workers   (scripts/rollout_worker.py, one frozen VLM each)
      \- 1 trainable action solver, in its own process (scripts/solver_server.py)

  ``theta`` (the solver's adapters and action head) lives inside the solver process and
  is updated there through the ``ActionSolver`` interface; ``phi`` (the pointer MLP) is
  a two-layer head and is updated here. Exactly one trainable solver process, so theta
  has a single owner: no parameter broadcast is needed and every rollout is on-policy
  with respect to the same theta. The workers are the scalable axis (CPU simulation
  plus the frozen encoder); they reload ``phi`` from disk whenever this process bumps
  its version.

MECHANISM BOUNDARY

  * Frozen: the VLM encoder, the plan extractor, the prototypes (stop-grad, reward
    only) and the BC reference policy. Trainable: exactly {phi, theta}.
  * The {stay, +1} monotone mask is never touched - both the rollout and the
    high-level gradient/KL go through ``PrimitiveClassifier``'s own masked candidates.
  * Alg.1 lines 40-45 write the update as an importance ratio r_i(Theta) against
    Theta_old. Exactly ONE gradient evaluation is taken per group of fresh rollouts, so
    r_i(Theta_old) == 1 identically and the ratio form degenerates to the
    advantage-weighted form of Eq. 9, which is what is implemented here
    (``grpo_awr.HighLevelUpdater`` plus the solver-side AWR step). Theta_old is the
    implicit snapshot taken at the start of every iteration.
  * Phi_t is computed by THIS process from the persisted shaping latents, not inside
    the rollout: nothing conditions on Phi, so the numbers are identical and the
    prototype bank keeps a single owner (see rl/rollout.py).
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from nsvla.config import Config
from nsvla.rl.pointer_metrics import group_pointer_metrics
from nsvla.utils import paths

REPO = Path(__file__).resolve().parents[2]
DEFAULT_POINTER = str(Path(paths.run_root()) / "pointer" / "pointer_config.json")


# --------------------------------------------------------------------------- #
# worker pool
# --------------------------------------------------------------------------- #
@dataclass
class WorkerPool:
    """N persistent rollout workers driven through the file protocol (see worker doc)."""

    jobs_dir: Path
    n_workers: int
    port: int
    clf_path: str
    suite: str
    gpus: list[str] = field(default_factory=lambda: ["0"])
    threads: int = 4
    nice: int = 10
    poll_s: float = 1.0
    log_dir: Path | None = None
    procs: list[subprocess.Popen] = field(default_factory=list)
    _job_seq: int = 0

    def start(self) -> None:
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        for k in range(self.n_workers):
            wdir = self.jobs_dir / f"w{k}"
            wdir.mkdir(parents=True, exist_ok=True)
            for stale in ("stop", "ready"):
                (wdir / stale).unlink(missing_ok=True)
            env = dict(os.environ)
            env["CUDA_VISIBLE_DEVICES"] = self.gpus[k % len(self.gpus)]
            env["MUJOCO_GL"] = "egl"
            for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                        "NUMEXPR_NUM_THREADS"):
                env[var] = str(self.threads)
            cmd = [
                "nice", "-n", str(self.nice), sys.executable,
                str(REPO / "scripts" / "rollout_worker.py"),
                "--worker-id", str(k), "--jobs-dir", str(self.jobs_dir),
                "--port", str(self.port), "--clf-path", self.clf_path,
                "--suite", self.suite, "--threads", str(self.threads),
            ]
            log = None
            if self.log_dir is not None:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                log = open(self.log_dir / f"worker{k}.log", "a")
            self.procs.append(subprocess.Popen(
                cmd, env=env, stdout=log or subprocess.DEVNULL,
                stderr=subprocess.STDOUT, start_new_session=True,
            ))
        print(f"[rl] launched {self.n_workers} rollout workers on GPUs {self.gpus}", flush=True)

    def wait_ready(self, timeout_s: float = 1800.0) -> None:
        t0 = time.time()
        while time.time() - t0 < timeout_s:
            ready = sum(1 for k in range(self.n_workers)
                        if (self.jobs_dir / f"w{k}" / "ready").exists())
            if ready == self.n_workers:
                print(f"[rl] all {ready} workers ready ({time.time()-t0:.0f}s)", flush=True)
                return
            self._check_alive()
            time.sleep(2.0)
        raise TimeoutError(f"only {ready}/{self.n_workers} workers became ready")

    def _check_alive(self) -> None:
        for k, p in enumerate(self.procs):
            if p.poll() is not None:
                raise RuntimeError(f"rollout worker {k} exited with code {p.returncode}; "
                                   f"see {self.log_dir}/worker{k}.log")

    def submit(self, jobs: list[dict], timeout_s: float = 7200.0) -> list[dict]:
        """Write one job per worker, block until all report ``.done``, return summaries."""
        self._job_seq += 1
        n = self._job_seq
        pending = []
        for k, job in enumerate(jobs):
            if not job["episodes"]:
                continue
            wdir = self.jobs_dir / f"w{k % self.n_workers}"
            path = wdir / f"job_{n}.json"
            path.write_text(json.dumps(job, indent=2))
            pending.append(path)
        out = []
        t0 = time.time()
        while pending:
            self._check_alive()
            still = []
            for p in pending:
                done = p.parent / (p.stem + ".done")
                err = p.parent / (p.stem + ".err")
                if done.exists():
                    out.append(json.loads(done.read_text()))
                elif err.exists():
                    raise RuntimeError(f"rollout job {p} failed:\n{err.read_text()}")
                else:
                    still.append(p)
            pending = still
            if pending:
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"rollout jobs timed out after {timeout_s}s: {pending}")
                time.sleep(self.poll_s)
        return out

    def stop(self) -> None:
        for k in range(self.n_workers):
            (self.jobs_dir / f"w{k}" / "stop").write_text("1")
        t0 = time.time()
        while time.time() - t0 < 60 and any(p.poll() is None for p in self.procs):
            time.sleep(1.0)
        for p in self.procs:
            if p.poll() is None:
                try:
                    os.killpg(os.getpgid(p.pid), signal.SIGTERM)
                except Exception:
                    pass


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class StageIITrainer:
    """Algorithm 1, end to end."""

    def __init__(
        self,
        cfg: Config,
        *,
        run_dir: Path,
        clf_path: str = DEFAULT_POINTER,
        task_ids: list[int] | None = None,
        worker_gpus: list[str] | None = None,
        port: int = 5678,
    ):
        import torch

        from nsvla.eval.run_suite import load_pointer
        from nsvla.rl.grpo_awr import HighLevelUpdater, RunningBaseline
        from nsvla.rl.prototypes import PrototypeBank
        from nsvla.rl.rewards import RewardShaper
        from nsvla.solver import make_solver

        self.cfg = cfg
        self.run_dir = Path(run_dir)
        self.suite = cfg.env.suites[0]
        self.task_ids = task_ids if task_ids is not None else list(range(cfg.env.tasks_per_suite))
        self.clf_path = clf_path
        for sub in ("traces", "ckpt", "logs", "jobs", "eval"):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        cfg.to_yaml(self.run_dir / "config.yaml")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # phi: the Stage-I BC pointer. The updater deep-copies it as the FROZEN BC
        # anchor pi^{h,*} at construction, so resume must load RL state AFTER this.
        clf, self.vocab = load_pointer(clf_path, self.device, suite=self.suite)
        self.updater = HighLevelUpdater(
            clf, lr=cfg.rl.lr, warmup_steps=cfg.rl.warmup_steps, schedule=cfg.rl.schedule,
            total_steps=cfg.rl.max_iters, beta_kl=cfg.rl.beta_kl, kl_band=tuple(cfg.rl.kl_band),
            use_kl_controller=cfg.rl.kl_controller, grad_clip=cfg.rl.grad_clip,
            device=self.device,
        )
        self.bank = PrototypeBank(
            n_primitives=len(self.vocab.ops), dim=cfg.classifier.d_in,
            buffer_cap=cfg.rl.prototype.buffer_cap, n_clusters=cfg.rl.prototype.n_clusters,
        )
        self.shaper = RewardShaper(cfg.rl.reward)
        self.baseline = RunningBaseline(cfg.rl.baseline_momentum)

        cfg.solver.port = port
        self.solver = make_solver(cfg.solver)
        self.pool = WorkerPool(
            jobs_dir=self.run_dir / "jobs", n_workers=cfg.rl.rollout.n_workers, port=port,
            clf_path=clf_path, suite=self.suite,
            gpus=worker_gpus or [os.environ.get("CUDA_VISIBLE_DEVICES", "0")],
            threads=cfg.rl.rollout.worker_threads, nice=cfg.rl.rollout.worker_nice,
            poll_s=cfg.rl.rollout.job_poll_s, log_dir=self.run_dir / "logs",
        )

        self.iteration = 0
        self.pointer_version = 0
        self.best_sr = -1.0
        self.no_improve = 0
        self.metrics_path = self.run_dir / "metrics.jsonl"
        self.rng = np.random.default_rng(cfg.seed)
        self._task_cycle: list[int] = []
        self._t_start = time.time()

    # ------------------------------------------------------------------ #
    # checkpointing / resume
    # ------------------------------------------------------------------ #
    def _pointer_ckpt_path(self) -> Path:
        return self.run_dir / "ckpt" / "pointer_latest.pt"

    def _publish_pointer(self) -> None:
        """Persist phi so workers pick it up on their next job (version-gated)."""
        import torch

        self.pointer_version += 1
        tmp = self._pointer_ckpt_path().with_suffix(".tmp")
        torch.save({
            "state_dict": self.updater.clf.state_dict(),
            "version": self.pointer_version,
        }, tmp)
        tmp.replace(self._pointer_ckpt_path())

    def save_state(self, tag: str | None = None) -> None:
        import torch

        ck = self.run_dir / "ckpt"
        torch.save(self.updater.state_dict(), ck / "updater.pt")
        torch.save(self.bank.state_dict(), ck / "prototypes.pt")
        (ck / "shaper.json").write_text(json.dumps({
            "shaper": self.shaper.state_dict(),
            "baseline": self.baseline.state_dict(),
        }, indent=2))
        (self.run_dir / "state.json").write_text(json.dumps({
            "iteration": self.iteration,
            "pointer_version": self.pointer_version,
            "best_sr": self.best_sr,
            "no_improve": self.no_improve,
            "rng": self.rng.bit_generator.state,
        }, indent=2, default=str))
        theta_dir = ck / (f"theta_{tag}" if tag else "theta_latest")
        try:
            # only the resume snapshot carries the AdamW state (disk discipline)
            self.solver.save_parameters(str(theta_dir), with_optim=(tag is None))
        except Exception as e:      # a non-trainable solver must not kill the run
            print(f"[rl] WARN save_parameters failed: {e!r}", flush=True)

    def load_state(self) -> bool:
        import torch

        sp = self.run_dir / "state.json"
        if not sp.exists():
            return False
        st = json.loads(sp.read_text())
        ck = self.run_dir / "ckpt"
        if (ck / "updater.pt").exists():
            self.updater.load_state_dict(torch.load(ck / "updater.pt", map_location=self.device))
        if (ck / "prototypes.pt").exists():
            self.bank.load_state_dict(torch.load(ck / "prototypes.pt", map_location="cpu"))
        if (ck / "shaper.json").exists():
            d = json.loads((ck / "shaper.json").read_text())
            self.shaper.load_state_dict(d["shaper"])
            self.baseline.load_state_dict(d["baseline"])
        if (ck / "theta_latest").exists():
            try:
                self.solver.load_parameters(str(ck / "theta_latest"))
            except Exception as e:
                print(f"[rl] WARN load_parameters failed: {e!r}", flush=True)
        self.iteration = int(st["iteration"])
        self.pointer_version = int(st.get("pointer_version", 0))
        self.best_sr = float(st.get("best_sr", -1.0))
        self.no_improve = int(st.get("no_improve", 0))
        print(f"[rl] resumed at iteration {self.iteration} (best_sr={self.best_sr:.3f})",
              flush=True)
        return True

    # ------------------------------------------------------------------ #
    def _next_task(self) -> int:
        """Alg.1 line 6: draw the instruction for this iteration.

        ``iid`` (the original) draws uniformly with replacement. ``cycle`` draws a fresh
        random permutation of the task list every |tasks| iterations, i.e. uniform
        WITHOUT replacement within a cycle. Both are unbiased samplers over the task
        distribution. ``cycle`` exists because a training trend is read by comparing an
        early window of iterations against a late one: with an iid draw the two windows
        cover different tasks, so a per-task difference in pointer behaviour can
        masquerade as a trend. ``cycle`` makes the windows task-matched by construction.
        It changes only which instruction is presented when; no part of the mechanism
        (Eq. 4/5/6/7/9) is touched.
        """
        order = getattr(self.cfg.rl, "task_order", "cycle")
        if order == "iid" or len(self.task_ids) <= 1:
            return int(self.task_ids[self.rng.integers(len(self.task_ids))])
        if not getattr(self, "_task_cycle", None):
            perm = self.rng.permutation(len(self.task_ids))
            self._task_cycle = [int(self.task_ids[i]) for i in perm]
        return self._task_cycle.pop(0)

    # ------------------------------------------------------------------ #
    # Alg.1 line 7-26: collect a group of rollouts
    # ------------------------------------------------------------------ #
    def collect(self, n_episodes: int, sigma: float, mode: str, task_id: int,
                out_dir: Path, ep_offset: int = 0) -> list[tuple]:
        from nsvla.rl.rollout import load_rollout

        n_w = self.pool.n_workers
        episodes: list[dict] = []
        for i in range(n_episodes):
            episodes.append({
                "rollout_id": f"it{self.iteration:04d}_{mode}_t{task_id}_g{i:02d}",
                "init_state_idx": ep_offset + i,
                "seed": ep_offset + i,
                "group_index": i,
            })
        jobs = []
        for k in range(n_w):
            jobs.append({
                "iteration": self.iteration, "mode": mode, "suite": self.suite,
                "task_id": task_id, "sigma": sigma, "out_dir": str(out_dir),
                "episodes": episodes[k::n_w],
                "high_sample": self.cfg.rl.rollout.high_sample and mode == "train",
                "pointer_ckpt": str(self._pointer_ckpt_path()),
                "pointer_version": self.pointer_version,
            })
        self.pool.submit(jobs, timeout_s=self.cfg.rl.rollout.job_timeout_s)
        rollouts = []
        for ep in episodes:
            tp = out_dir / f"{ep['rollout_id']}.json"
            rollouts.append(load_rollout(tp))
        return rollouts

    # ------------------------------------------------------------------ #
    # Alg.1 lines 17, 22-23: potentials and shaped rewards
    # ------------------------------------------------------------------ #
    def _grasped(self, probe: dict) -> float:
        """Grasp signal fed to r^sub.

        ``stock`` = the trace's ``probe['grasped']`` flag (grip_gap < 0.03, calibrated
        on thin rims). ``recal`` = ``obj_lift > 0.02 and grip_gap < 0.070``, whose recall
        against the simulator's own grasp predicate is 1.000 on all four suites. The
        default is ``recal``: the stock flag misses most real grasps on box-shaped
        objects, which would zero out those tasks' dense sub-goal reward and would also
        score r^sub on a different criterion from the pointer metrics.
        """
        from nsvla.rl.pointer_metrics import recal_grasped

        if getattr(self.cfg.rl.reward, "grasp_criterion", "recal") == "stock":
            return float((probe or {}).get("grasped", 0.0))
        return recal_grasped(probe)

    def shape_rewards(self, trace, feats) -> dict:
        from nsvla.rl.rewards import gripper_alignment, subgoal_indicator

        T = len(trace.steps)
        ell = feats["ell"]                       # (T+1, d)
        prim = feats["prim_id"]                  # (T,)
        # Phi_t = -min_c ||ell_t - mu_{m_t,c}||^2 ; Phi_T = 0 on a terminal (App. C.1)
        phis = np.zeros(T + 1, dtype=np.float64)
        for t in range(min(T, ell.shape[0])):
            phis[t] = self.bank.potential_np(int(prim[t]), ell[t])
        if T + 1 <= ell.shape[0]:
            phis[T] = 0.0 if trace.success else self.bank.potential_np(
                int(prim[T - 1]) if T else 0, ell[T]
            )

        ops = [p.get("op", "") for p in trace.plan]
        r_sub = np.zeros(T, dtype=np.float64)
        r_gr = np.zeros(T, dtype=np.float64)
        for t, s in enumerate(trace.steps):
            op = ops[min(s.m_t, len(ops) - 1)] if ops else ""
            r_sub[t] = subgoal_indicator(op, self._grasped(s.probe))
            r_gr[t] = gripper_alignment(op, s.probe.get("grip_gap", 1.0))

        b = np.array([s.b_t for s in trace.steps], dtype=np.float64)
        r_task = 1.0 if trace.success else 0.0
        out = self.shaper.episode(
            phis=phis, b_flags=b, r_task=r_task, r_sub=r_sub, r_gr=r_gr,
            actions=feats.get("exec_chunk"),
        )
        out["phis"] = phis
        return out

    # ------------------------------------------------------------------ #
    # Alg.1 lines 28-35: successful-segment buffers
    # ------------------------------------------------------------------ #
    def update_buffers(self, trace, feats) -> int:
        """Insert per-primitive segment summaries ell_bar from SUCCESSFUL rollouts.

        Alg.1 line 32 gates on ``b_te = 1 and r_task_te > 0``. In LIBERO the sparse
        task reward can only fire at the terminal decision, so read literally the rule
        would admit at most the final segment of a successful episode. We take the
        intended reading — every completed segment of a **successful** rollout is a
        successful segment (the pointer is monotone, so segments are contiguous and
        ordered, App. C.1 "Boundary extraction") — and record the choice here because
        it is the only place where the pseudocode is ambiguous.
        """
        import torch

        if not trace.success:
            return 0
        T = len(trace.steps)
        if T == 0:
            return 0
        ell = feats["ell"]
        prim = feats["prim_id"]
        m = np.array([s.m_t for s in trace.steps], dtype=np.int64)
        inserted = 0
        start = 0
        for t in range(1, T + 1):
            if t == T or m[t] != m[start]:
                seg = ell[start:t]
                if seg.shape[0]:
                    self.bank.add_segment(
                        int(prim[start]), torch.as_tensor(seg.mean(axis=0))
                    )
                    inserted += 1
                start = t
        return inserted

    # ------------------------------------------------------------------ #
    # Alg.1 lines 37-45: the bi-granular update
    # ------------------------------------------------------------------ #
    def update_high(self, rollouts, shaped) -> dict:
        import torch

        psi, m_prev, taken, ridx, plan_ids, plan_len = [], [], [], [], [], []
        for i, ((trace, feats), sh) in enumerate(zip(rollouts, shaped)):
            T = len(trace.steps)
            if T == 0:
                continue
            psi.append(feats["psi"])
            m_prev.append(feats["m_prev"])
            taken.append(feats["taken"])
            ridx.append(np.full(T, i, dtype=np.int64))
            plan_ids.append(np.tile(feats["plan_op_ids"], (T, 1)))
            plan_len.append(np.full(T, int(feats["plan_len"][0]), dtype=np.int64))
        if not psi:
            return {"n_decisions": 0}
        returns = torch.tensor([sh["R_high"] for sh in shaped], dtype=torch.float32)
        return self.updater.update(
            psi=torch.as_tensor(np.concatenate(psi), dtype=torch.float32),
            plan_op_ids=torch.as_tensor(np.concatenate(plan_ids), dtype=torch.long),
            m_prev=torch.as_tensor(np.concatenate(m_prev), dtype=torch.long),
            plan_len=torch.as_tensor(np.concatenate(plan_len), dtype=torch.long),
            taken_slot=torch.as_tensor(np.concatenate(taken), dtype=torch.long),
            rollout_index=torch.as_tensor(np.concatenate(ridx), dtype=torch.long),
            returns=returns,
        )

    def update_low(self, rollouts, shaped) -> dict:
        import torch

        items, advs = [], []
        for (trace, feats), sh in zip(rollouts, shaped):
            r_low = sh["r_low"]
            for t, s in enumerate(trace.steps):
                if s.sample_id is None or t >= feats["normalized"].shape[0]:
                    continue
                items.append({
                    "sample_id": s.sample_id,
                    "target": feats["normalized"][t],
                    "reward": float(r_low[t]) if t < len(r_low) else 0.0,
                })
                advs.append(float(r_low[t]) if t < len(r_low) else 0.0)
        if not items:
            return {"n": 0}
        rew = torch.tensor(advs, dtype=torch.float32)
        self.baseline.update(rew)
        adv = self.baseline.advantages(rew, clip=self.cfg.rl.adv_clip)

        idx = np.arange(len(items))
        cap = self.cfg.rl.awr_max_chunks
        if cap and len(idx) > cap:      # uniform subsample: capping by advantage would bias AWR
            idx = self.rng.choice(idx, size=cap, replace=False)
            idx.sort()
        batch = [{"sample_id": items[j]["sample_id"],
                  "target": items[j]["target"],
                  "advantage": float(adv[j])} for j in idx]
        out = self.solver.awr_update(
            batch, beta_l=self.cfg.rl.beta_l, lr=self.cfg.rl.low_lr
        )
        # write the per-step advantages back into the traces (trace = only metric source)
        pos = 0
        for (trace, feats), sh in zip(rollouts, shaped):
            for t, s in enumerate(trace.steps):
                if s.sample_id is None or t >= feats["normalized"].shape[0]:
                    continue
                s.r_low = items[pos]["reward"]
                s.adv_low = float(adv[pos])
                pos += 1
        out["adv_low_mean"] = float(adv.mean())
        out["adv_low_std"] = float(adv.std(unbiased=False))
        out["n_chunks_total"] = len(items)
        return out

    # ------------------------------------------------------------------ #
    def persist_traces(self, rollouts, shaped, iter_dir: Path) -> None:
        """Rewrite each trace with its reward and advantage fields filled in."""
        for (trace, feats), sh in zip(rollouts, shaped):
            phis = sh["phis"]
            for t, s in enumerate(trace.steps):
                s.phi = float(phis[t]) if t < len(phis) else 0.0
                s.r_seg = float(self.cfg.rl.reward.lambda_seg * s.b_t)
                s.r_task = float(1.0 if (trace.success and t == len(trace.steps) - 1) else 0.0)
                s.r_low_terms = {k: float(v[t]) if t < len(v) else 0.0
                                 for k, v in sh["norm"].items()}
            trace.rl = dict(trace.rl or {})
            trace.rl.update({
                "R_high": sh["R_high"],
                "return_low": float(np.sum(sh["r_low"])),
            })
            trace.save(iter_dir / f"{trace.rl['rollout_id']}.json")

    # ------------------------------------------------------------------ #
    def eval_probe(self, n_episodes: int) -> dict:
        """Greedy probe on the current Theta; SR comes from the traces, never inline."""
        out_dir = self.run_dir / "eval" / f"iter{self.iteration:04d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        srs, n = [], 0
        per_task_sr: dict[str, float] = {}
        per_task_ptr: dict[str, dict] = {}
        for task_id in self.task_ids:
            per_task = max(1, n_episodes // max(1, len(self.task_ids)))
            rolls = self.collect(per_task, self.cfg.rl.rollout.sigma_eval, "eval",
                                 task_id, out_dir, ep_offset=1000)
            hits = [1.0 if tr.success else 0.0 for tr, _ in rolls]
            srs += hits
            n += len(rolls)
            # per-task breakdown, so the probe can be stratified by plan length M
            # (the M=1 / M>=2 split, along which the pointer has nothing to learn or
            # something to learn respectively)
            per_task_sr[str(task_id)] = float(np.mean(hits)) if hits else 0.0
            pt = group_pointer_metrics([tr for tr, _ in rolls])
            pt.pop("ptr_evidence", None)
            per_task_ptr[str(task_id)] = pt
        sr = float(np.mean(srs)) if srs else 0.0
        m1 = [t for t in self.task_ids if per_task_ptr.get(str(t), {}).get("ptr_plan_len", 0) <= 1]
        mge2 = [t for t in self.task_ids if per_task_ptr.get(str(t), {}).get("ptr_plan_len", 0) > 1]
        sr_m1 = float(np.mean([per_task_sr[str(t)] for t in m1])) if m1 else None
        sr_mge2 = float(np.mean([per_task_sr[str(t)] for t in mge2])) if mge2 else None
        (out_dir / "probe.json").write_text(json.dumps(
            {"iteration": self.iteration, "SR": sr, "n_episodes": n,
             "per_task_SR": per_task_sr, "per_task_pointer": per_task_ptr,
             "SR_M1": sr_m1, "SR_Mge2": sr_mge2,
             "tasks_M1": m1, "tasks_Mge2": mge2}, indent=2, default=float))
        return {"eval_SR": sr, "eval_n": n, "eval_SR_M1": sr_m1, "eval_SR_Mge2": sr_mge2}

    # ------------------------------------------------------------------ #
    def run(self, max_iters: int | None = None, resume: bool = True) -> dict:
        cfg = self.cfg
        max_iters = max_iters if max_iters is not None else cfg.rl.max_iters
        if resume:
            self.load_state()
        self._publish_pointer()          # Alg.1 line 2: Theta_old <- Theta
        self.pool.start()
        try:
            self.pool.wait_ready()
            ping = self.solver.ping()
            print(f"[rl] solver ping: {ping}", flush=True)
            if not ping.get("trainable"):
                raise RuntimeError(
                    "the action solver is not trainable; start it with --trainable")
            return self._loop(max_iters)
        finally:
            self.pool.stop()

    def _loop(self, max_iters: int) -> dict:
        cfg = self.cfg
        history: list[dict] = []
        stop_reason = "max_iters"
        while self.iteration < max_iters:
            it_t0 = time.time()
            self.iteration += 1
            iter_dir = self.run_dir / "traces" / f"iter{self.iteration:04d}"
            iter_dir.mkdir(parents=True, exist_ok=True)

            # Alg.1 line 5: (optional) refresh prototypes every U iterations
            refreshed = False
            if self.iteration % max(1, cfg.rl.prototype.refresh_every) == 0:
                self.bank.refresh()
                refreshed = True

            # Alg.1 line 6: sample an instruction (== a task of the suite)
            task_id = self._next_task()
            ep_offset = int(self.rng.integers(0, 40)) * cfg.rl.group_size

            # Alg.1 lines 7-26
            rollouts = self.collect(cfg.rl.group_size, cfg.rl.rollout.sigma, "train",
                                    task_id, iter_dir, ep_offset=ep_offset)
            shaped = [self.shape_rewards(tr, ft) for tr, ft in rollouts]
            # Alg.1 lines 28-35 (buffers are filled AFTER the rewards of this group are
            # computed, so a group is scored under the prototypes it was collected with)
            n_seg = sum(self.update_buffers(tr, ft) for tr, ft in rollouts)

            # Alg.1 lines 37-45
            hi = self.update_high(rollouts, shaped)
            lo = self.update_low(rollouts, shaped)
            self._publish_pointer()      # Alg.1 line 46: Theta_old <- Theta
            self.persist_traces(rollouts, shaped, iter_dir)

            R = np.array([s["R_high"] for s in shaped], dtype=np.float64)
            rec = {
                "iteration": self.iteration,
                "task_id": task_id,
                "wall_s": round(time.time() - it_t0, 1),
                "rollout_SR": float(np.mean([1.0 if tr.success else 0.0 for tr, _ in rollouts])),
                "R_high_mean": float(R.mean()), "R_high_std": float(R.std()),
                "mean_episode_len": float(np.mean([tr.episode_len for tr, _ in rollouts])),
                "mean_decisions": float(np.mean([len(tr.steps) for tr, _ in rollouts])),
                "n_segments_buffered": int(n_seg),
                "proto_buffer": self.bank.n_filled(),
                "proto_refreshed": refreshed,
                "phi_mean": float(np.mean([s["phis"][:-1].mean() if len(s["phis"]) > 1 else 0.0
                                           for s in shaped])),
                "loss_joint": float(hi.get("loss_high", 0.0)
                                    + cfg.rl.lambda_l * lo.get("loss_low", 0.0)),
                **hi, **lo,
            }
            # what the pointer DID inside this group of rollouts, scored with the
            # recalibrated grasp criterion (see rl/pointer_metrics.py).
            pm = group_pointer_metrics([tr for tr, _ in rollouts])
            evidence = pm.pop("ptr_evidence", None)
            rec.update(pm)
            if evidence is not None:
                (iter_dir / "pointer_evidence.json").write_text(
                    json.dumps(evidence, indent=2, default=float))

            if self.iteration % max(1, cfg.rl.eval_every) == 0:
                rec.update(self.eval_probe(cfg.rl.eval_episodes))
                if rec["eval_SR"] > self.best_sr + 1e-9:
                    self.best_sr = rec["eval_SR"]
                    self.no_improve = 0
                    self.save_state(tag="best")
                else:
                    self.no_improve += 1
            if self.iteration % max(1, cfg.rl.ckpt_every) == 0:
                self.save_state()

            with self.metrics_path.open("a") as f:
                f.write(json.dumps(rec, default=float) + "\n")
            history.append(rec)
            print(f"[rl] iter {self.iteration}: SR={rec['rollout_SR']:.2f} "
                  f"R_high={rec['R_high_mean']:.3f}+-{rec['R_high_std']:.3f} "
                  f"KL={rec.get('kl', 0):.2e} beta={rec.get('beta_kl', 0):.3f} "
                  f"L_high={rec.get('loss_high', 0):.4f} L_low={rec.get('loss_low', 0):.4f} "
                  f"({rec['wall_s']}s)", flush=True)

            if self.no_improve >= cfg.rl.early_stop_patience:
                stop_reason = "sr_plateau"
                break
            if (time.time() - self._t_start) / 3600.0 > cfg.rl.wall_clock_h_per_suite:
                stop_reason = "wall_clock"
                break

        self.save_state(tag="final")
        summary = {
            "iterations": self.iteration,
            "best_eval_SR": self.best_sr,
            "stop_reason": stop_reason,
            "history": history,
        }
        (self.run_dir / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
        return summary


# --------------------------------------------------------------------------- #
def train_rl(
    cfg: Config,
    *,
    clf_path: str = DEFAULT_POINTER,
    task_ids: list[int] | None = None,
    worker_gpus: list[str] | None = None,
    port: int = 5678,
    max_iters: int | None = None,
    resume: bool = True,
) -> dict:
    """Run Stage-II H-GRPO + AWR optimization for one experiment (Alg.1)."""
    run_dir = Path(cfg.run_root) / cfg.exp_name
    trainer = StageIITrainer(
        cfg, run_dir=run_dir, clf_path=clf_path, task_ids=task_ids,
        worker_gpus=worker_gpus, port=port,
    )
    return trainer.run(max_iters=max_iters, resume=resume)
