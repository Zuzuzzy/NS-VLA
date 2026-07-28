"""Stage-II rollout collection — Alg.1 lines 7-26 (one episode per call).

This is the RL twin of ``eval.run_suite.rollout_episode`` and deliberately shares
its skeleton (same plan extraction, same monotone pointer call, same rendered
sub-instruction, same env probe, same trace schema). The three differences are all
Alg.1 requirements:

  * line 18 - the chunk is **sampled** (``ActionSolver.act_sample``: deterministic
    head plus Gaussian dither in normalized action space) instead of taken greedily;
  * lines 12/16 - one frozen VLM forward per decision yields BOTH the pointer feature
    ``psi_bar_t`` (Eq. 5) and the shaping latent ``ell_t = E_w(o_t)``;
  * lines 21/24 - the per-step tensors the update needs (psi, ell, executed normalized
    chunk, pointer state, sample handle) are persisted next to the trace.

WHAT IS *NOT* DONE HERE, ON PURPOSE: the potential ``Phi_t`` (line 17), the shaped
reward (line 23) and the advantages (line 37) are computed by the master from the
persisted ``ell``. Nothing in the rollout *conditions* on Phi — the policy never
sees it — so computing it afterwards is mathematically identical to computing it
inline, and it keeps prototype state in ONE process (no broadcast to N workers) and
keeps every reward number reproducible offline from the trace + npz.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from nsvla.config import Config
from nsvla.eval.traces import EpisodeTrace, Step, hash_action_chunk


def _sample_pointer_step(clf, psi_t, plan_op_ids, m_prev, plan_len, temperature: float, rng):
    """Optional stochastic pointer (ablation). Same masked {stay,+1} set as Eq. 5.

    Sampling happens INSIDE the admissible set, so Proposition 1 (monotonicity, at
    most M ordered segments) is untouched: only which admissible slot is taken.
    Alg.1 line 13 uses argmax, which is the default (``rollout.high_sample=False``).
    """
    import torch
    import torch.nn.functional as F

    with torch.no_grad():
        vocab_logits = clf.logits(psi_t)
        cand, dup = clf._candidates(m_prev, plan_len)
        cand_ops = torch.gather(plan_op_ids, 1, cand)
        cand_logits = torch.gather(vocab_logits, 1, cand_ops).clone()
        cand_logits[:, 1] += clf.advance_bias
        cand_logits[dup, 1] = float("-inf")
        probs = F.softmax(cand_logits / max(temperature, 1e-6), dim=1)
        u = float(rng.random())
        choice = torch.tensor([1 if u < float(probs[0, 1]) else 0], device=probs.device)
        m_t = torch.gather(cand, 1, choice[:, None]).squeeze(1)
        b_t = (m_t != m_prev).long()
        logp = torch.log(torch.gather(probs, 1, choice[:, None]).squeeze(1) + 1e-9)
    return m_t, b_t, logp


def rl_rollout_episode(
    cfg: Config,
    env,
    task,
    suite: str,
    comp,
    *,
    init_state,
    seed: int,
    rollout_id: str,
    sigma: float,
    max_steps: int,
    iteration: int = 0,
    group_index: int = 0,
    mode: str = "train",
    verbose: bool = False,
) -> tuple[EpisodeTrace, dict[str, np.ndarray]]:
    """Collect ONE rollout under the current (Theta_old) policy. Alg.1 lines 8-26."""
    import torch

    from nsvla.encoder.plan_extractor import extract_plan
    from nsvla.eval.run_suite import _pick_target, _probe_env
    from nsvla.solver.bridge import render_sub_instruction

    # Seed material is hashed, not passed through python hash(): the builtin is salted
    # per interpreter unless PYTHONHASHSEED is pinned, which would make a multi-process
    # run unreplayable.
    _seed_material = f"{cfg.seed}|{rollout_id}|{iteration}".encode()
    rng = np.random.default_rng(
        int(hashlib.md5(_seed_material).hexdigest()[:8], 16)
    )
    instruction = task.language
    obs = env.reset(task, init_state=init_state)
    for _ in range(env.NUM_STEPS_WAIT):           # object-settle wait
        obs, _, _, _ = env.step(env.DUMMY_ACTION)

    target = _pick_target(env)
    tgt_pos = obs["raw"].get(f"{target}_pos") if target else None
    z0 = float(tgt_pos[2]) if tgt_pos is not None else None

    trace = EpisodeTrace(
        task_id=task.bddl_file, suite=suite, seed=seed, instruction=instruction
    )
    plan = extract_plan(instruction, comp.vocab, max_len=cfg.plan.max_len)
    trace.plan = plan.to_list()
    op_ids_list = plan.padded_op_ids(comp.vocab)
    plan_op_ids = torch.tensor([op_ids_list], device=comp.device)
    plan_len_t = torch.tensor([max(1, plan.M)], device=comp.device)
    m_prev = torch.zeros(1, dtype=torch.long, device=comp.device)

    psis: list[np.ndarray] = []
    ells: list[np.ndarray] = []
    normalized: list[np.ndarray] = []
    m_prev_hist: list[int] = []
    taken_hist: list[int] = []
    prim_ids: list[int] = []
    exec_chunks: list[np.ndarray] = []

    t0 = time.time()
    t = 0
    decision = 0
    success = False
    obs_last = obs
    while t < max_steps:
        probe = _probe_env(obs, target, z0)
        # Alg.1 lines 12 and 16: ONE frozen forward -> pointer feature AND shaping latent
        psi, ell = comp.encoder.context_and_shaping(obs["agentview_pointer"], instruction)
        psi_t = torch.as_tensor(psi, dtype=torch.float32, device=comp.device).unsqueeze(0)

        prev_val = int(m_prev.item())
        if mode == "train" and cfg.rl.rollout.high_sample:
            m_t, b_t, logp = _sample_pointer_step(
                comp.clf, psi_t, plan_op_ids, m_prev, plan_len_t,
                cfg.rl.rollout.high_temperature, rng,
            )
        else:                                     # Alg.1 line 13: greedy Eq. 5 inference
            m_t, b_t, logp = comp.clf.step(psi_t, plan_op_ids, m_prev, plan_len_t)
        m_prev = m_t
        m_val, b_val = int(m_t.item()), int(b_t.item())
        sub_instr = render_sub_instruction(plan, m_val, instruction)

        # Alg.1 line 18: sample the chunk (sigma=0 at eval-probe time => greedy)
        out = comp.solver.act_sample(
            obs["full_image"], sub_instr, obs["state"],
            wrist_image=obs["wrist_image"], sigma=sigma,
            seed=int(rng.integers(0, 2**31 - 1)),
        )
        chunk = out["action_chunk"]

        for a in chunk:                            # Alg.1 line 20: execute open-loop
            obs, _r, done, _info = env.step(a)
            t += 1
            if done:
                success = True
                break
            if t >= max_steps:
                break
        obs_last = obs

        psis.append(psi)
        ells.append(ell)
        normalized.append(out["normalized_chunk"])
        exec_chunks.append(chunk)
        m_prev_hist.append(prev_val)
        taken_hist.append(m_val)
        prim_ids.append(int(op_ids_list[min(m_val, len(op_ids_list) - 1)]))

        trace.add_step(Step(
            t=decision, m_t=m_val, b_t=b_val,
            action_chunk_hash=hash_action_chunk(chunk),
            sub_instr=sub_instr, probe=probe,
            logp=float(logp.item()),
            sample_id=out["sample_id"],
        ))
        decision += 1
        if success:
            break

    # Alg.1 line 21: the boundary observation AFTER the last chunk closes the
    # telescoping sum; on a terminal we use the absorbing convention Phi_T = 0
    # (App. C.1), which the master applies — here we only record ell_T.
    _, ell_T = comp.encoder.context_and_shaping(obs_last["agentview_pointer"], instruction)
    ells.append(ell_T)

    trace.success = success
    trace.episode_len = t
    trace.wall_time = time.time() - t0
    trace.rl = {
        "iteration": iteration,
        "group_index": group_index,
        "rollout_id": rollout_id,
        "sigma": float(sigma),
        "mode": mode,
        "r_task": 1.0 if success else 0.0,
        "n_decisions": decision,
    }
    feats = {
        "psi": np.asarray(psis, dtype=np.float32),                 # (T, d)
        "ell": np.asarray(ells, dtype=np.float32),                 # (T+1, d)
        "normalized": np.asarray(normalized, dtype=np.float32),    # (T, H, action_dim)
        "exec_chunk": np.asarray(exec_chunks, dtype=np.float32),   # (T, H, action_dim)
        "m_prev": np.asarray(m_prev_hist, dtype=np.int64),         # (T,)
        "taken": np.asarray(taken_hist, dtype=np.int64),           # (T,)
        "prim_id": np.asarray(prim_ids, dtype=np.int64),           # (T,)
        "plan_op_ids": np.asarray(op_ids_list, dtype=np.int64),    # (Mmax,)
        "plan_len": np.asarray([max(1, plan.M)], dtype=np.int64),
    }
    if verbose:
        ptr = [s.m_t for s in trace.steps]
        print(f"  [rl:{mode}] {suite}/{Path(task.bddl_file).stem[:36]} rid={rollout_id} "
              f"success={success} len={t} decisions={decision} pointer={ptr}", flush=True)
    return trace, feats


def save_rollout(trace: EpisodeTrace, feats: dict[str, np.ndarray], out_dir: str | Path,
                 name: str) -> dict[str, str]:
    """Write ``<name>.json`` (trace) + ``<name>.npz`` (update tensors)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    npz = out_dir / f"{name}.npz"
    np.savez_compressed(npz, **feats)
    if trace.rl is None:
        trace.rl = {}
    trace.rl["feature_path"] = str(npz)
    js = out_dir / f"{name}.json"
    trace.save(js)
    return {"trace": str(js), "features": str(npz)}


def load_rollout(trace_path: str | Path) -> tuple[EpisodeTrace, dict[str, np.ndarray]]:
    """Inverse of ``save_rollout`` (master side)."""
    trace = EpisodeTrace.load(trace_path)
    feats: dict[str, Any] = {}
    fp = (trace.rl or {}).get("feature_path")
    if fp and Path(fp).exists():
        with np.load(fp) as z:
            feats = {k: z[k] for k in z.files}
    return trace, feats
