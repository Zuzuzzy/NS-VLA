"""Evaluation rollout harness (NS-VLA §5.1).

Rolls a policy over the evaluation matrix (suites x tasks x episodes on the
benchmark's reproducible initial-state set) and writes ONE trace per episode
(``eval.traces``); metrics are computed separately (``eval.metrics``), never here.
Two policies, for a like-for-like comparison:

  * ``nsvla``  - the full NS-VLA decision loop:
        episode start: extract_plan (frozen rule parser) -> fixed plan p;
        each decision step (H-chunk boundary):
            psi_bar = VLM(o_t, x)                            (encoder.vlm_features)
            m_t, b_t = pointer.step(psi_bar, plan, m_{t-1})  (Eq. 5, {stay, +1})
            x_tilde  = render_sub_instruction(p, m_t, x)     (solver.bridge)
            chunk    = solver.act(o_t, x_tilde, S_t)         (Eq. 7, H-step open loop)
  * ``solver_bare`` - the action solver alone: the full instruction is sent at every
        decision step, with no plan and no pointer (m_t = 0, b_t = 0).

Both drive the SAME action solver and emit the SAME trace schema, so the metrics are
policy-agnostic and the two arms differ only in what the solver is conditioned on.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from nsvla.config import Config
from nsvla.eval.traces import EpisodeTrace, Step, hash_action_chunk
from nsvla.utils import paths
from nsvla.utils import paths

DEFAULT_CLF_PATH = str(Path(paths.run_root()) / "pointer" / "clf.pt")


# --------------------------------------------------------------------------- #
# component construction
# --------------------------------------------------------------------------- #
@dataclass
class Components:
    solver: Any                  # ActionSolver
    encoder: Any = None          # VLMFeatureEncoder (nsvla policy only)
    clf: Any = None              # PrimitiveClassifier (nsvla policy only)
    vocab: Any = None
    device: str = "cpu"


def load_pointer(clf_path: str, device: str, suite: str | None = None):
    """Load the trained monotone-pointer classifier + its vocab.

    ``clf_path`` is either a ``clf.pt`` or a ``pointer_config.json`` naming one. Beyond
    the weights, the checkpoint or config may carry the static advance-logit calibration
    constant: a scalar ``advance_bias`` and/or a per-suite map
    ``advance_bias_per_suite``, from which ``suite`` selects, falling back to the scalar.
    Nothing here touches the decision rule; with both fields absent the bias is 0.0.
    """
    import json
    from pathlib import Path

    import torch

    from nsvla.encoder.primitive_clf import PrimitiveClassifier
    from nsvla.primitives.vocab import PrimitiveVocab, default_vocab

    bias_map: dict = {}
    bias_default: float | None = None
    p = Path(clf_path)
    if p.suffix == ".json":
        cfg = json.loads(p.read_text())
        bias_map = dict(cfg.get("advance_bias_per_suite", {}))
        bias_default = cfg.get("advance_bias_default")
        ck = Path(cfg["ckpt"])
        clf_path = str(ck if ck.is_absolute() else p.parent / ck)

    ckpt = torch.load(clf_path, map_location=device)
    if not bias_map:
        bias_map = dict(ckpt.get("advance_bias_per_suite", {}))
    if bias_default is None:
        bias_default = float(ckpt.get("advance_bias", 0.0))
    advance_bias = float(bias_map.get(suite, bias_default)) if suite else float(bias_default)

    clf = PrimitiveClassifier(
        d_in=int(ckpt["d_in"]),
        vocab_size=int(ckpt["vocab_size"]),
        hidden=int(ckpt["hidden"]),
        # architecture and calibration constants come from the checkpoint; absent =>
        # 2 layers and a zero bias, i.e. the uncalibrated decision rule.
        n_layers=int(ckpt.get("n_layers", 2)),
        advance_bias=advance_bias,
    )
    clf.load_state_dict(ckpt["state_dict"])
    clf.eval().to(device)
    # keep the map so run_suite can switch the constant when a run spans several suites
    clf.advance_bias_per_suite = bias_map
    clf.advance_bias_default = bias_default
    # ckpt stored the full op list incl. <pad>/<noop>; strip specials for the ctor
    if "ops" in ckpt:
        base = [o for o in list(ckpt["ops"]) if o not in ("<pad>", "<noop>")]
        vocab = PrimitiveVocab(base)
    else:
        vocab = default_vocab()
    return clf, vocab


def build_components(
    cfg: Config, policy: str, clf_path: str = DEFAULT_CLF_PATH, device: str | None = None
) -> Components:
    import torch

    from nsvla.solver import make_solver

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    comp = Components(solver=make_solver(cfg.solver), device=device)
    if policy == "nsvla":
        from nsvla.encoder.vlm_features import VLMFeatureEncoder

        # a run usually targets one suite; hand it to the loader so a per-suite
        # calibration constant resolves right away (run_suite re-selects per suite below)
        suite0 = cfg.env.suites[0] if len(cfg.env.suites) == 1 else None
        comp.clf, comp.vocab = load_pointer(clf_path, device, suite=suite0)
        comp.encoder = VLMFeatureEncoder(image_size=cfg.control.image_size)
        comp.encoder.load()
    return comp


# --------------------------------------------------------------------------- #
# physical grasp probe (diagnostics only — never feeds the policy)
# --------------------------------------------------------------------------- #
# Panda finger separation is bimodal: ~0.079 fully open, ~0.004-0.005 clamped on a
# thin rim, so 0.03 sits in the empty middle. Closure alone cannot distinguish
# "holding an object" from "closed on air", so a grasp additionally requires the pick
# target to have left its resting height.
GRIP_CLOSED_GAP = 0.03      # metres
OBJ_LIFT_THRESH = 0.02      # metres above the object's post-settle resting z


def _pick_target(env) -> str | None:
    """Name of the object the task's first primitive picks (``obj_of_interest[0]``)."""
    try:
        objs = env._env.obj_of_interest
        return str(objs[0]) if objs else None
    except Exception:
        return None


def _probe_env(obs: dict, target: str | None, z0: float | None) -> dict[str, float]:
    """{grip_gap, obj_lift, grasped, eef_z} for one decision step (diagnostic)."""
    state = obs["state"]
    gap = float(abs(state[6] - state[7]))       # gripper_qpos = state[6:8]
    out: dict[str, float] = {"grip_gap": gap, "eef_z": float(state[2])}
    lift = 0.0
    if target is not None and z0 is not None:
        pos = obs["raw"].get(f"{target}_pos")
        if pos is not None:
            lift = float(pos[2]) - z0
    out["obj_lift"] = lift
    out["grasped"] = float(gap < GRIP_CLOSED_GAP and lift > OBJ_LIFT_THRESH)
    return out


# --------------------------------------------------------------------------- #
# single-episode rollout
# --------------------------------------------------------------------------- #
def rollout_episode(
    cfg: Config,
    env,
    task,
    suite: str,
    seed: int,
    init_state,
    policy: str,
    comp: Components,
    max_steps: int,
    verbose: bool = False,
) -> EpisodeTrace:
    """Run one episode; return its EpisodeTrace. ``policy`` in {nsvla, solver_bare}."""
    import torch

    from nsvla.encoder.plan_extractor import extract_plan
    from nsvla.solver.bridge import render_sub_instruction

    instruction = task.language
    obs = env.reset(task, init_state=init_state)

    # object-settle wait (benchmark convention: fixed dummy steps before acting)
    for _ in range(env.NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(env.DUMMY_ACTION)

    # grasp-probe baseline: pick target's resting height AFTER the settle wait
    target = _pick_target(env)
    tgt_pos = obs["raw"].get(f"{target}_pos") if target else None
    z0 = float(tgt_pos[2]) if tgt_pos is not None else None

    trace = EpisodeTrace(
        task_id=task.bddl_file, suite=suite, seed=seed, instruction=instruction
    )

    # NS-VLA: fixed plan + monotone pointer state
    plan = plan_op_ids = plan_len = m_prev = None
    if policy == "nsvla":
        plan = extract_plan(instruction, comp.vocab, max_len=cfg.plan.max_len)
        trace.plan = plan.to_list()
        plan_op_ids = torch.tensor([plan.padded_op_ids(comp.vocab)], device=comp.device)
        plan_len = torch.tensor([max(1, plan.M)], device=comp.device)
        m_prev = torch.zeros(1, dtype=torch.long, device=comp.device)

    t0 = time.time()
    t = 0                # policy env steps executed (excludes the settle wait)
    decision = 0
    success = False
    while t < max_steps:
        # probe the observation the pointer is about to decide ON (pre-chunk), so a
        # "pointer advanced without a grasp" claim is evaluated at the decision moment
        probe = _probe_env(obs, target, z0)
        if policy == "nsvla":
            # pointer view (vflip) matches the cached training-feature orientation
            psi = comp.encoder.context(obs["agentview_pointer"], instruction)
            psi_t = torch.as_tensor(psi, dtype=torch.float32, device=comp.device).unsqueeze(0)
            m_t, b_t, _ = comp.clf.step(psi_t, plan_op_ids, m_prev, plan_len)
            m_prev = m_t
            m_val, b_val = int(m_t.item()), int(b_t.item())
            sub_instr = render_sub_instruction(plan, m_val, instruction)
        else:  # solver_bare
            m_val, b_val, sub_instr = 0, 0, instruction

        chunk = comp.solver.act(
            obs["full_image"], sub_instr, obs["state"], wrist_image=obs["wrist_image"]
        )

        # execute the H-step chunk open-loop
        for a in chunk:
            obs, _r, done, _info = env.step(a)
            t += 1
            if done:
                success = True
                break
            if t >= max_steps:
                break

        trace.add_step(Step(
            t=decision, m_t=m_val, b_t=b_val,
            action_chunk_hash=hash_action_chunk(chunk),
            sub_instr=sub_instr,
            probe=probe,
        ))
        decision += 1
        if success:
            break

    trace.success = success
    trace.episode_len = t
    trace.wall_time = time.time() - t0
    if verbose:
        ptr = [s.m_t for s in trace.steps] if policy == "nsvla" else None
        print(f"  [{policy}] {suite}/{Path(task.bddl_file).stem[:40]} ep_seed={seed} "
              f"success={success} len={t} decisions={decision} pointer={ptr}", flush=True)
    return trace


# --------------------------------------------------------------------------- #
# suite driver
# --------------------------------------------------------------------------- #
def run_suite(
    cfg: Config,
    *,
    policy: str | None = None,
    n_tasks: int | None = None,
    n_episodes: int | None = None,
    task_ids: list[int] | None = None,
    clf_path: str = DEFAULT_CLF_PATH,
    device: str | None = None,
    comp: Components | None = None,
    verbose: bool = True,
    task_meta: dict[int, dict] | None = None,
) -> list[EpisodeTrace]:
    """Roll out ``policy`` over cfg.env.suites; write one trace/episode; return them.

    ``n_tasks`` / ``n_episodes`` cap the matrix for quick runs (default: full).
    ``task_ids`` selects an explicit task subset and overrides ``n_tasks``; this is how
    a single (arm, task) unit is sharded to one worker process, and how a LIBERO-Plus
    unit passes its sampled perturbation tasks.
    ``comp`` lets a caller share a loaded encoder and solver across policies.
    ``task_meta`` is {task_id: {"perturb_axis": str, "perturb_level": int}} and only
    stamps those two trace fields (LIBERO-Plus); it never influences the rollout.
    """
    from nsvla.envs.libero_env import LiberoEnv, select_libero_env

    # MUST run before the `from libero.libero import benchmark` below: LIBERO and
    # LIBERO-Plus are two trees exporting the same `libero` package, selected by
    # sys.path + LIBERO_CONFIG_PATH, and the first import wins for the process.
    select_libero_env(cfg.env)

    policy = policy or cfg.eval.policy
    run_dir = Path(cfg.run_root) / cfg.exp_name
    traces_dir = run_dir / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)

    if comp is None:
        comp = build_components(cfg, policy, clf_path=clf_path, device=device)

    from libero.libero import benchmark
    bench = benchmark.get_benchmark_dict()

    all_traces: list[EpisodeTrace] = []
    for suite in cfg.env.suites:
        # per-suite static calibration constant; a constant only, the decision rule
        # below is untouched. With no map, the scalar already set stands.
        bmap = getattr(comp.clf, "advance_bias_per_suite", None) if comp.clf is not None else None
        if bmap:
            comp.clf.advance_bias = float(
                bmap.get(suite, getattr(comp.clf, "advance_bias_default", comp.clf.advance_bias)))
            if verbose:
                print(f"[pointer] {suite}: advance_bias={comp.clf.advance_bias:+.3f}", flush=True)
        suite_obj = bench[suite]()
        env = LiberoEnv(cfg.env, image_size=cfg.control.image_size)
        max_steps = env.max_steps(suite)
        if task_ids is not None:
            ids = [i for i in task_ids if 0 <= i < suite_obj.n_tasks]
        else:
            n_t = suite_obj.n_tasks if n_tasks is None else min(n_tasks, suite_obj.n_tasks)
            ids = list(range(n_t))
        n_ep = cfg.eval.episodes_per_task if n_episodes is None else n_episodes
        try:
            for task_id in ids:
                task = suite_obj.get_task(task_id)
                init_states = suite_obj.get_task_init_states(task_id)
                for ep in range(n_ep):
                    init_state = init_states[ep % len(init_states)]
                    trace = rollout_episode(
                        cfg, env, task, suite, seed=ep, init_state=init_state,
                        policy=policy, comp=comp, max_steps=max_steps, verbose=verbose,
                    )
                    meta = (task_meta or {}).get(task_id)
                    if meta:
                        trace.perturb_axis = meta.get("perturb_axis")
                        trace.perturb_level = meta.get("perturb_level")
                    fname = f"{suite}__{Path(task.bddl_file).stem}__ep{ep}__{policy}.json"
                    trace.save(traces_dir / fname)
                    all_traces.append(trace)
        finally:
            env.close()
    return all_traces
