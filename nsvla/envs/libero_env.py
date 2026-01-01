"""Unified LIBERO / LIBERO-Plus env wrapper (NS-VLA §5.1).

One interface, two benchmarks, switched by ``EnvConfig.benchmark``:
  * ``"LIBERO"``      -> in-distribution 4 suites (1-shot main result);
  * ``"LIBERO-Plus"`` -> seven-axis perturbation suite (OOD, eval only).
The two live in different source trees, so the correct one is prepended to
``sys.path`` at construction (``select_libero_pythonpath``) — they expose the same
``libero`` package name and must not both be importable at once.

Contract (shared by the eval harness and the RL rollout):
  reset(task, perturb=None) -> obs{image 224x224x3 uint8, proprio [d] float, ...}
  step(action[-1,1]^7)      -> (obs, reward, done, info)

Requires the LIBERO simulator (robosuite / mujoco) at run time; the path-selection
logic above is importable without it.
"""
from __future__ import annotations

import math
import os
import sys
from typing import Any

import numpy as np

from nsvla.config import EnvConfig

# Benchmark -> (source root providing the `libero` package, LIBERO_CONFIG_PATH dir).
_BENCHMARK_ATTRS = {
    "LIBERO": ("libero_root", "libero_config_dir"),
    "LIBERO-Plus": ("libero_plus_root", "libero_plus_config_dir"),
}
# accepted spellings on the CLI / in configs
BENCHMARK_ALIASES = {
    "libero": "LIBERO", "LIBERO": "LIBERO",
    "libero_plus": "LIBERO-Plus", "libero-plus": "LIBERO-Plus",
    "LIBERO-Plus": "LIBERO-Plus", "LIBERO_PLUS": "LIBERO-Plus",
}


def canon_benchmark(name: str) -> str:
    try:
        return BENCHMARK_ALIASES[str(name)]
    except KeyError:
        raise ValueError(
            f"unknown benchmark {name!r}; expected one of {sorted(set(BENCHMARK_ALIASES.values()))}"
        ) from None


def select_libero_env(cfg: EnvConfig) -> tuple[str, str]:
    """Point this PROCESS at one benchmark's LIBERO tree. -> (source root, config dir).

    LIBERO and LIBERO-Plus are two source trees that both export the package name
    ``libero``, and each also needs its OWN ``config.yaml`` (assets / bddl_files /
    init_states roots). Two things are therefore set, and both must happen BEFORE the
    first ``import libero`` in the process:

      * ``sys.path``            -- chosen root first, the other root removed, so exactly
                                   one tree is importable;
      * ``LIBERO_CONFIG_PATH``  -- a per-benchmark directory holding that tree's
                                   ``config.yaml`` (``libero.libero.__init__`` reads the
                                   env var at import time and re-reads the file on every
                                   ``get_libero_path`` call).

    Using a SECOND config dir rather than rewriting ``~/.libero/config.yaml`` is
    deliberate: several evaluation processes run at once, and a write-then-restore of
    the single shared file would silently repoint every concurrently-running
    standard-LIBERO process at LIBERO-Plus assets. The standard installation is left
    untouched: no uninstall, no ``pip install -e`` of LIBERO-Plus, no edit of the
    shared config file.
    """
    bench = canon_benchmark(cfg.benchmark)
    root_attr, cfgdir_attr = _BENCHMARK_ATTRS[bench]
    chosen = getattr(cfg, root_attr)
    other = getattr(cfg, _BENCHMARK_ATTRS["LIBERO-Plus" if bench == "LIBERO" else "LIBERO"][0])
    # Drop the other benchmark's root if present, then put the chosen one first.
    sys.path[:] = [p for p in sys.path if p != other]
    if chosen in sys.path:
        sys.path.remove(chosen)
    sys.path.insert(0, chosen)

    cfgdir = getattr(cfg, cfgdir_attr, None) or os.path.expanduser("~/.libero")
    os.environ["LIBERO_CONFIG_PATH"] = cfgdir

    already = sys.modules.get("libero")
    if already is not None and getattr(already, "__file__", None):
        if not os.path.abspath(already.__file__).startswith(os.path.abspath(chosen) + os.sep):
            raise RuntimeError(
                f"`libero` is already imported from {already.__file__} but benchmark "
                f"{bench!r} needs {chosen}; the two trees cannot coexist in one process "
                f"-- select the benchmark before importing libero (or use a new process)."
            )
    return chosen, cfgdir


def select_libero_pythonpath(cfg: EnvConfig) -> str:
    """Back-compat shim: ``select_libero_env`` returning only the source root."""
    return select_libero_env(cfg)[0]


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """robosuite (x,y,z,w) quaternion -> axis-angle exp-coords (matches VLA-Adapter eval)."""
    quat = np.asarray(quat, dtype=np.float64).copy()
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


class LiberoEnv:
    """Thin wrapper over a LIBERO task env with the eval-harness observation contract.

    Rendering / preprocessing matches the reference VLA-Adapter evaluation: the sim is
    built at ``env_img_res`` (256), images are rotated 180 degrees (``[::-1, ::-1]``) and
    proprio is ``[eef_pos(3), quat2axisangle(eef_quat)(3), gripper_qpos(2)]`` (8-dim).
    The 256->224 policy resize and centre crop happen solver-side, so the raw render is
    passed over the wire verbatim.

    The obs dict exposes BOTH orientations, for the two consumers:
      * ``agentview`` / ``wrist``        - RAW render, feeding the frozen VLM features
        that the pointer classifier was trained on;
      * ``full_image`` / ``wrist_image`` - ROTATED, feeding the action solver.
    """

    # Reference TASK_MAX_STEPS, plus the object-settle wait.
    MAX_STEPS = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    NUM_STEPS_WAIT = 10
    DUMMY_ACTION = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0]

    def __init__(self, cfg: EnvConfig, image_size: int = 224, env_img_res: int = 256, seed: int = 0):
        self.cfg = cfg
        self.image_size = image_size      # policy input size (resize happens solver-side)
        self.env_img_res = env_img_res    # sim render size (official env_img_res)
        self.seed = seed
        self.libero_root, self.libero_config_dir = select_libero_env(cfg)
        self._env = None            # underlying OffScreenRenderEnv, built lazily per task
        self._env_task_key = None   # bddl of the currently-built env (reuse across episodes)

    # ------------------------------------------------------------------ #
    def _build_env(self, task: Any) -> None:
        import os

        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        if self._env is not None and self._env_task_key == bddl:
            return
        self.close()
        env = OffScreenRenderEnv(
            bddl_file_name=bddl,
            camera_heights=self.env_img_res,
            camera_widths=self.env_img_res,
        )
        env.seed(self.seed)  # affects object positions; keep fixed across arms
        self._env = env
        self._env_task_key = bddl

    def _hide_collision_geoms(self) -> None:
        """Render only visual meshes (match the official VLA-Adapter / vla-env rendering).

        The official eval renders with mujoco ``vopt.geomgroup = [0,1,1,0,0,0]`` (collision
        group 0 hidden). On mujoco 2.3.7 the python ``geomgroup`` setter is broken (writing
        any element zeroes the whole array), so robosuite's ``render_collision_mesh=False``
        silently no-ops and coloured collision geoms render over the robot, pushing the
        solver inputs out of distribution. Making every group-0 geom transparent (alpha=0)
        is bit-identical to the official render at an identical sim state. Re-applied on
        every reset because a hard reset restores the XML rgba defaults.
        """
        try:
            model = self._env.env.sim.model
            for i in range(model.ngeom):
                if int(model.geom_group[i]) == 0:
                    model.geom_rgba[i][3] = 0.0
        except Exception:  # never let a render tweak break the rollout
            pass

    def _obs_dict(self, obs: dict) -> dict:
        agent = np.asarray(obs["agentview_image"])
        wrist = np.asarray(obs["robot0_eye_in_hand_image"])
        state = np.concatenate((
            np.asarray(obs["robot0_eef_pos"], dtype=np.float64),
            _quat2axisangle(obs["robot0_eef_quat"]),
            np.asarray(obs["robot0_gripper_qpos"], dtype=np.float64),
        )).astype(np.float32)
        return {
            "agentview": agent,                 # raw live render
            "wrist": wrist,                     # raw live render
            # Pointer Qwen view: vertically flipped to match the cached training
            # features, which were extracted from hdf5 ``obs/agentview_rgb`` (stored
            # in the opposite vertical convention to the live robosuite obs). The
            # orientation matters: fed the raw view, the pointer never leaves slot 0.
            "agentview_pointer": agent[::-1],
            "full_image": agent[::-1, ::-1],    # rotated 180 (solver primary view)
            "wrist_image": wrist[::-1, ::-1],   # rotated 180 (solver wrist view)
            "state": state,                     # (8,) proprio
            "raw": obs,
        }

    def reset(self, task: Any, init_state: Any = None, perturb: dict | None = None) -> dict:
        """Reset to ``task`` at a benchmark initial state -> obs dict.

        ``init_state`` is one row of ``task_suite.get_task_init_states(task_id)``
        (the benchmark's reproducible initial-state set).

        ``perturb`` stays None for BOTH benchmarks. Under LIBERO-Plus a perturbation is
        not a runtime knob: it is baked into the task identity itself. The seven axes are
        encoded in the task NAME, which LIBERO-Plus's ``ControlEnv.__init__`` parses out of
        the bddl path (``..._view_<h>_<v>_<scale>_<rot>_<vert>_initstate_<k>[_noise_<n>]``
        -> camera pose / robot variant / sensor corruption) or which resolves to a
        dedicated bddl on disk (``_table_<n>`` background, ``_light_<n>`` lighting,
        ``_add_<n>`` layout, ``_language_<n>`` instruction rewrite). Selecting the
        perturbation therefore means selecting the TASK, and the axis/level of the task
        that was run is recorded in the trace from ``benchmark/task_classification.json``.
        """
        if perturb is not None:
            raise NotImplementedError(
                "LIBERO-Plus perturbations are selected by task id, not by a reset kwarg")
        self._build_env(task)
        self._env.reset()
        if init_state is not None:
            obs = self._env.set_init_state(init_state)
        else:
            obs = self._env.reset()
        self._hide_collision_geoms()  # match official rendering (see method docstring)
        return self._obs_dict(obs)

    def step(self, action) -> tuple[dict, float, bool, dict]:
        """Apply a 7-dim action in [-1, 1] -> (obs, reward, done, info)."""
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, reward, done, info = self._env.step(a.tolist())
        return self._obs_dict(obs), float(reward), bool(done), info

    def max_steps(self, suite: str) -> int:
        return self.MAX_STEPS.get(suite, 300)

    def close(self) -> None:
        if self._env is not None:
            self._env.close()
            self._env = None
            self._env_task_key = None
