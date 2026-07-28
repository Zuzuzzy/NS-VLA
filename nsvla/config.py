"""Experiment configuration with YAML round-trip.

One experiment is one config is one run directory. Every value here is a knob; the
mechanism (Eq. 4/5/6/7/8/9) does not depend on any of them.

NOTE: deliberately NOT using ``from __future__ import annotations`` - the YAML
round-trip (``_fromdict``) reflects over ``dataclasses.fields(...).type`` and needs
real class objects, not stringized annotations.
"""
import dataclasses
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from nsvla.primitives.vocab import DEFAULT_OPS
from nsvla.utils import paths


@dataclass
class ControlConfig:
    """Observation / action / chunk control.

    H must match the action solver's own chunk length, or the baseline comparison is
    not like-for-like.
    """

    chunk_H: int = 8            # action chunk length (VLA-Adapter LIBERO default)
    action_dim: int = 7        # 6-DoF delta + gripper
    image_size: int = 224
    proprio_dim: int = 8
    action_clip: float = 1.0   # actions clipped to [-1, 1]
    normalize_proprio: bool = True


@dataclass
class PlanConfig:
    """Plan extraction (Eq. 4).

    Mmax=6 is a harmless cap. Under the Fig. 4a operation-level expansion (every
    transport clause -> pick + place_*), the longest LIBERO plan is a two-clause
    LIBERO-Long transport = 4 primitives (pick,place,pick,place); 6 leaves margin.
    """

    max_len: int = 6            # Mmax (>= observed max of 4 across all 40 LIBERO tasks)
    ops: list[str] = field(default_factory=lambda: list(DEFAULT_OPS))
    use_vlm_fallback: bool = True   # rule parser first, VLM only for unresolved slots


@dataclass
class ClassifierConfig:
    """Monotone pointer classifier g_phi (Eq. 5). Two-layer MLP, hidden = mult * d_in."""

    d_in: int = 2048           # frozen-VLM mean-pooled feature dim
    hidden_mult: float = 2.0   # hidden = round(hidden_mult * d_in)
    seg_window: int = 3        # segment-end window w in {2,3,4}, chosen on held-out frames
    class_weight_inverse_freq: bool = True  # inverse-frequency class weights (imbalance only)

    @property
    def hidden(self) -> int:
        return int(round(self.hidden_mult * self.d_in))


@dataclass
class SparsifyConfig:
    """Primitive-conditioned token sparsification (App. F). Proportional top-p ~ K=32."""

    top_p: float = 0.125       # fraction of tokens kept (~12.5% ~= 32 of 256)
    soft_temperature: float = 0.5   # soft top-p gate temperature at train time
    hard_at_inference: bool = True  # hard top-p selection at inference
    k_floor: int = 1           # always keep at least this many tokens


@dataclass
class BridgeConfig:
    """Primitive-conditioned bridging (Eq. 6). Two configurable forms."""

    mode: str = "render"       # "embed": e_t = Wu Embed(u,arg)+Wpsi psi+WS S ; "render": rendered sub-instruction x_tilde
    embed_dim: int = 512       # d_u for the op-arg embedding stream
    controller_dim: int = 512  # d_e controller dimension


@dataclass
class SolverConfig:
    """Which action solver to use and how to reach it (``nsvla.solver``).

    ``name`` selects a registered ``ActionSolver``. ``remote`` runs the solver in its
    own process behind ZMQ, which is how the reference VLA-Adapter solver is used (its
    checkpoints need a different dependency set from the encoder side); ``vla_adapter``
    loads it in-process; ``fake`` is the deterministic CPU stand-in used by the tests.
    Fields not understood by a given solver are ignored.
    """

    name: str = "remote"            # "remote" | "vla_adapter" | "fake"
    checkpoint: str = ""            # solver weights; empty => the solver's own default
    unnorm_key: str = "libero_spatial_no_noops"
    task_suite: str = "libero_spatial"
    use_pro_version: bool = False
    source_root: str = ""           # solver source tree; empty => NSVLA_SOLVER_ROOT
    chunk_H: int = 8
    action_dim: int = 7
    host: str = "127.0.0.1"
    port: int = 5678
    request_timeout_ms: int = 60000
    device: str = "cuda"
    dtype: str = "bfloat16"


@dataclass
class PrototypeConfig:
    """Prototype shaping (App. C, Alg.1). Feeds the reward only (stop-grad)."""

    encoder: str = "vlm_vision"       # frozen shaping encoder E_w
    buffer_cap: int = 64        # per-primitive FIFO successful-segment buffer cap
    n_clusters: int = 3         # C cluster centres per primitive
    refresh_every: int = 5      # refresh prototypes every U iterations
    stop_grad: bool = True      # prototypes are constant within an update (mechanism)


@dataclass
class RewardConfig:
    """Bi-granular reward weights (Eq. 8). Components are running-std normalized to the
    same scale, then scaled by priority coefficients milestone:progress:aux = 1:0.5:0.1."""

    gamma: float = 0.99        # shaping discount for the telescoping potential
    lambda_seg: float = 1.0    # high-level: milestone / boundary weight
    lambda_prog: float = 1.0   # low-level: progress (potential telescoping)
    lambda_sub: float = 0.5    # low-level: sub-goal reaching
    lambda_sm: float = 0.1     # low-level: action smoothness penalty
    lambda_gr: float = 0.1     # low-level: gripper-state alignment
    normalize_running_std: bool = True
    # Grasp criterion behind r^sub. "recal" = obj_lift > 0.02 and grip_gap < 0.070,
    # which has recall 1.000 against the simulator's own grasp predicate on all four
    # suites; "stock" = the trace's probe['grasped'] flag, whose tighter gap threshold
    # misses box-shaped objects entirely. See rl/pointer_metrics.py.
    grasp_criterion: str = "recal"


@dataclass
class RolloutConfig:
    """Stage-II rollout collection (Alg.1 lines 7-26).

    N environment-worker processes (CPU simulation + the frozen VLM on GPU) share ONE
    solver process per GPU. Exploration is the Gaussian chunk noise of Alg.1 line 18
    (``sigma``, in NORMALIZED action space, i.e. the [-1, 1] pre-unnormalization scale
    the regression head works in), so a deterministic head becomes a stochastic policy
    without needing a density: AWR only needs the executed chunk, not its log-prob.
    """

    n_workers: int = 4          # env worker processes
    sigma: float = 0.05         # Gaussian exploration std on the normalized chunk
    sigma_eval: float = 0.0     # greedy at eval-probe time
    high_sample: bool = False   # Alg.1 line 13 infers m_hat by argmax; sampling is an ablation
    high_temperature: float = 1.0
    worker_threads: int = 4     # per-process thread cap; unthrottled workers thrash
    worker_nice: int = 10
    job_poll_s: float = 1.0
    job_timeout_s: float = 3600.0
    store_features: bool = True  # persist psi_bar / ell per decision step for the update


@dataclass
class RLConfig:
    """Stage-II optimization (Eq. 9 / Alg.1)."""

    lr: float = 3e-5           # phi = pointer MLP
    low_lr: float = 1e-5       # theta = the solver's adapters + action head
    # Measured in ITERATIONS, not minibatches: the high-level factor takes exactly ONE
    # optimizer step per Alg.1 iteration, so a 100-step warmup would still sit at a
    # tenth of the target learning rate when a 50-iteration run ends.
    warmup_steps: int = 5
    schedule: str = "cosine"
    group_size: int = 8        # G rollouts per group (raise to 16 if 16-core parallel rollout)
    beta_kl: float = 0.05      # KL anchor coeff to BC reference
    kl_band: tuple[float, float] = (1e-3, 5e-2)   # target KL band; out-of-band => adjust
    kl_controller: bool = True # adapt beta_kl to keep KL inside the band
    beta_l: float = 1.0        # chunk temperature (AWR low-level weight)
    lambda_l: float = 1.0      # mixing of low-level loss into joint objective
    baseline_momentum: float = 0.9   # running baseline for the chunk-level advantage
    adv_clip: float = 5.0      # clip normalized advantages (App. C.1 numerical stability)
    awr_micro_bs: int = 1      # the reference solver's forward is batch-1
    awr_max_chunks: int = 64   # cap chunk samples per AWR update (wall-clock guard)
    awr_epochs: int = 1
    max_iters: int = 50        # not hard-fixed; early-stop on 10-eval SR plateau
    early_stop_patience: int = 10
    wall_clock_h_per_suite: float = 24.0
    grad_clip: float = 1.0
    bf16: bool = True
    grad_checkpoint: bool = True
    # "cycle" = uniform without replacement inside each pass over the task list,
    # "iid" = uniform with replacement. See StageIITrainer._next_task.
    task_order: str = "cycle"
    eval_every: int = 5        # eval probe cadence (iterations)
    eval_episodes: int = 10    # episodes per eval probe
    ckpt_every: int = 5
    reward: RewardConfig = field(default_factory=RewardConfig)
    prototype: PrototypeConfig = field(default_factory=PrototypeConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)


@dataclass
class EnvConfig:
    """LIBERO / LIBERO-Plus environment selection (``nsvla.envs.libero_env``)."""

    benchmark: str = "LIBERO"   # "LIBERO" | "LIBERO-Plus"
    libero_root: str = field(default_factory=paths.libero_root)
    libero_plus_root: str = field(default_factory=paths.libero_plus_root)
    # Per-benchmark LIBERO_CONFIG_PATH dir (holds config.yaml: assets / bddl /
    # init_states roots). Two directories rather than rewriting the single shared
    # config, so a LIBERO-Plus process can never repoint a concurrently running
    # standard-LIBERO process.
    libero_config_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/.libero"))
    libero_plus_config_dir: str = field(
        default_factory=lambda: os.path.expanduser("~/.libero_plus"))
    suites: list[str] = field(
        default_factory=lambda: ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    )
    tasks_per_suite: int = 10
    perturb_axis: str | None = None   # LIBERO-Plus seven-axis perturbation (eval only)
    perturb_level: int | None = None


@dataclass
class EvalConfig:
    """Evaluation harness: identical seeds and initial states across the compared arms."""

    episodes_per_task: int = 50
    seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    policy: str = "nsvla"       # "nsvla" | "vla_adapter_bare"


@dataclass
class Config:
    """Top-level config composing every sub-config. ``from_yaml`` / ``to_yaml`` round-trip."""

    exp_name: str = "nsvla_debug"
    run_root: str = field(default_factory=paths.run_root)
    seed: int = 0
    control: ControlConfig = field(default_factory=ControlConfig)
    plan: PlanConfig = field(default_factory=PlanConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    sparsify: SparsifyConfig = field(default_factory=SparsifyConfig)
    bridge: BridgeConfig = field(default_factory=BridgeConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    rl: RLConfig = field(default_factory=RLConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    # ---- serialization -------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    def to_yaml(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        return _fromdict(cls, d)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open() as f:
            d = yaml.safe_load(f) or {}
        return cls.from_dict(d)

    @property
    def run_dir(self) -> Path:
        return Path(self.run_root) / self.exp_name


def _asdict(obj: Any) -> Any:
    """dataclasses.asdict but converting tuples to lists for clean YAML."""

    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _asdict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _asdict(v) for k, v in obj.items()}
    return obj


def _fromdict(cls: type, d: dict[str, Any]) -> Any:
    """Recursively rebuild a (possibly nested) dataclass from a plain dict.

    Unknown keys are ignored; missing keys keep dataclass defaults. Fields typed
    as a nested dataclass are rebuilt recursively; tuple-typed fields are coerced
    back from the YAML list form.
    """
    if not (dataclasses.is_dataclass(cls) and isinstance(cls, type)):
        return d
    kwargs: dict[str, Any] = {}
    field_map = {f.name: f for f in dataclasses.fields(cls)}
    for name, f in field_map.items():
        if name not in d:
            continue
        val = d[name]
        ftype = f.type
        if dataclasses.is_dataclass(ftype) and isinstance(val, dict):
            kwargs[name] = _fromdict(ftype, val)
        elif _is_tuple_type(ftype) and isinstance(val, list):
            kwargs[name] = tuple(val)
        else:
            kwargs[name] = val
    return cls(**kwargs)


def _is_tuple_type(ftype: Any) -> bool:
    origin = getattr(ftype, "__origin__", None)
    return origin is tuple or ftype is tuple


def default_config() -> Config:
    return Config()
