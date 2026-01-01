"""Reference action solver: VLA-Adapter (NS-VLA §4.2, Eq. 7).

This is the concrete ``ActionSolver`` the paper's numbers were produced with. It is
one implementation of the interface, not a dependency of the framework: swapping it
for another VLA means writing another ``ActionSolver`` and registering it.

The heavy path calls the upstream VLA-Adapter evaluation pipeline verbatim
(``run_libero_eval.initialize_model`` + ``robot_utils.get_action`` +
``run_libero_eval.process_action``), so the actions returned are what upstream would
execute for the same rotated images, proprio and instruction. NS-VLA conditioning
enters through the ``instruction`` argument alone: the solver receives the rendered
sub-instruction of the active primitive.

Upstream's checkpoints load under an older ``transformers`` than this repository
targets, so in practice this solver runs inside the VLA-Adapter environment behind
``nsvla.solver.server``, and the encoder side talks to it through
``nsvla.solver.remote.RemoteSolver`` (``solver.name: remote``). Running it in-process
is supported and needs no code change when the two dependency sets are compatible.

Reproducing the released results also needs three changes to the upstream tree: select
SDPA attention so no flash-attn build is required, register the LIBERO 1-shot and
sub-instruction dataset variants, and seed every RNG the fine-tuning path draws from
while guaranteeing a checkpoint at ``max_steps``.
"""
from __future__ import annotations

import sys
from typing import Any

import numpy as np

from nsvla.solver.base import ActionSolver, register_solver
from nsvla.utils import paths


@register_solver("vla_adapter")
class VLAAdapterSolver(ActionSolver):
    """VLA-Adapter policy behind the ``ActionSolver`` interface (lazy model load)."""

    def __init__(
        self,
        cfg: Any = None,
        checkpoint: str | None = None,
        unnorm_key: str = "libero_spatial_no_noops",
        task_suite: str = "libero_spatial",
        use_pro_version: bool = False,
        use_minivlm: bool = True,
        chunk_H: int = 8,
        action_dim: int = 7,
        source_root: str | None = None,
        trainable: bool = False,
        freeze_theta: bool = False,
        lora_rank: int = 64,
        lora_target: str = "all-linear",
        low_lr: float = 1e-5,
        grad_clip: float = 1.0,
        sample_cache: int = 4096,
        sample_seed: int = 0,
        max_vram_mb: int = 0,
    ):
        if cfg is not None:
            checkpoint = getattr(cfg, "checkpoint", checkpoint)
            unnorm_key = getattr(cfg, "unnorm_key", unnorm_key)
            task_suite = getattr(cfg, "task_suite", task_suite)
            use_pro_version = getattr(cfg, "use_pro_version", use_pro_version)
            chunk_H = getattr(cfg, "chunk_H", chunk_H)
            action_dim = getattr(cfg, "action_dim", action_dim)
            source_root = getattr(cfg, "source_root", source_root)
        self.checkpoint = checkpoint or paths.model_root() + "/vla-adapter"
        self.unnorm_key = unnorm_key
        self.task_suite = task_suite
        self.use_pro_version = use_pro_version
        self.use_minivlm = use_minivlm
        self.chunk_H = chunk_H
        self.action_dim = action_dim
        self.source_root = source_root or paths.solver_root()
        self.trainable = trainable
        self.freeze_theta = freeze_theta
        self.lora_rank = lora_rank
        self.lora_target = lora_target
        self.low_lr = low_lr
        self.grad_clip = grad_clip
        self.sample_cache = sample_cache
        self.max_vram_mb = max_vram_mb
        self._rng = np.random.default_rng(sample_seed)

        self._cfg = None
        self._predict = None
        self._trainer = None

    # ------------------------------------------------------------------ #
    # model loading
    # ------------------------------------------------------------------ #
    def _add_source_root(self) -> None:
        """Put the VLA-Adapter tree on the path, checking it is actually there first.

        Without this check the failure surfaces as ``ModuleNotFoundError: experiments``
        several frames deeper, which says nothing about what to install or where.
        """
        import os

        if not os.path.isdir(self.source_root):
            raise FileNotFoundError(
                f"VLA-Adapter source tree not found at {self.source_root!r}. Clone it, "
                f"then point NSVLA_SOLVER_ROOT (or --source-root) at it."
            )
        if self.source_root not in sys.path:
            sys.path.insert(0, self.source_root)

    def _build_generate_config(self):
        self._add_source_root()
        from experiments.robot.libero.run_libero_eval import GenerateConfig

        return GenerateConfig(
            model_family="openvla",
            pretrained_checkpoint=self.checkpoint,
            use_l1_regression=True,
            use_minivlm=self.use_minivlm,
            use_film=False,
            num_images_in_input=2,
            use_proprio=True,
            center_crop=True,
            num_open_loop_steps=self.chunk_H,
            unnorm_key=self.unnorm_key,
            task_suite_name=self.task_suite,
            use_pro_version=self.use_pro_version,
            save_version="vla-adapter",
        )

    def _cap_vram(self) -> None:
        """Bound this process's CUDA allocator when the card is shared with a trainer."""
        if not self.max_vram_mb:
            return
        import torch

        if not torch.cuda.is_available():
            return
        total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        frac = min(1.0, self.max_vram_mb / total_mb)
        torch.cuda.set_per_process_memory_fraction(frac, 0)
        print(f"[vla-adapter] CUDA allocator capped at {self.max_vram_mb} MiB "
              f"({frac:.3f} of {total_mb:.0f} MiB); the CUDA context sits on top",
              flush=True)

    def _install_unnorm_key_fallback(self) -> None:
        """Accept checkpoints whose norm_stats key is not ``<suite>[_no_noops]``.

        Upstream replaces ``model.norm_stats`` with the checkpoint's own
        ``dataset_statistics.json`` and then asserts that it contains the task-suite
        key. A checkpoint fine-tuned on a differently named dataset (for instance a
        1-shot split) carries exactly one key under that dataset's name, and the
        assertion fires before the server can start. The wrapper runs the upstream
        check first and only acts if it raises, so any checkpoint upstream already
        accepts is unaffected; on failure it falls back to the configured key, or to
        the sole key when there is exactly one.
        """
        self._add_source_root()
        from experiments.robot.libero import run_libero_eval as rle

        upstream = rle.check_unnorm_key
        if getattr(upstream, "_nsvla_wrapped", False):
            return
        configured = self.unnorm_key

        def check_unnorm_key(cfg, model):
            try:
                upstream(cfg, model)
                return
            except AssertionError as e:
                stats = getattr(model, "norm_stats", {}) or {}
                keys = list(stats)
                if configured in stats:
                    chosen, why = configured, "configured unnorm_key"
                elif len(keys) == 1:
                    chosen, why = keys[0], "sole norm_stats entry"
                else:
                    raise AssertionError(
                        f"{e}; unnorm_key {configured!r} is not among {keys}") from e
                cfg.unnorm_key = chosen
                print(f"[vla-adapter] unnorm_key fallback -> {chosen!r} ({why}; "
                      f"available: {keys})", flush=True)

        check_unnorm_key._nsvla_wrapped = True
        rle.check_unnorm_key = check_unnorm_key

    def load(self) -> None:
        """Load weights and build the greedy prediction closure."""
        if self._predict is not None:
            return
        cfg = self._build_generate_config()
        self._cap_vram()
        self._install_unnorm_key_fallback()
        from experiments.robot.libero.run_libero_eval import initialize_model, process_action
        from experiments.robot.robot_utils import get_action, set_seed_everywhere

        set_seed_everywhere(cfg.seed)
        (model, action_head, proprio_projector,
         noisy_action_projector, processor) = initialize_model(cfg)
        print(f"[vla-adapter] loaded {cfg.pretrained_checkpoint} "
              f"(use_pro_version={cfg.use_pro_version}, unnorm_key={cfg.unnorm_key})",
              flush=True)

        if self.trainable:
            from nsvla.solver.vla_adapter_trainer import VLAAdapterTrainer

            self._trainer = VLAAdapterTrainer(
                cfg, model, action_head, proprio_projector, processor,
                lora_rank=self.lora_rank, lora_target=self.lora_target,
                lr=self.low_lr, grad_clip=self.grad_clip,
                sample_cache_cap=self.sample_cache, freeze=self.freeze_theta,
            )
            model = self._trainer.vla
            print(f"[vla-adapter] trainable: {self._trainer.info()}", flush=True)

        def predict(full_image, wrist_image, instruction, proprio):
            obs = {
                "full_image": np.ascontiguousarray(full_image),
                "wrist_image": np.ascontiguousarray(wrist_image),
                "state": np.asarray(proprio, dtype=np.float64),
            }
            actions = get_action(
                cfg, model, obs, instruction,
                processor=processor,
                action_head=action_head,
                proprio_projector=proprio_projector,
                noisy_action_projector=noisy_action_projector,
                use_film=cfg.use_film,
                use_minivlm=cfg.use_minivlm,
            )
            processed = [process_action(np.asarray(a), cfg.model_family) for a in actions]
            return np.stack(processed, axis=0).astype(np.float32)

        self._cfg = cfg
        self._predict = predict
        self.chunk_H = cfg.num_open_loop_steps

    # ------------------------------------------------------------------ #
    # ActionSolver
    # ------------------------------------------------------------------ #
    def act(self, image, instruction, proprio, wrist_image=None) -> np.ndarray:
        if self._predict is None:
            self.load()
        return self._predict(
            image, image if wrist_image is None else wrist_image, instruction, proprio)

    @property
    def supports_training(self) -> bool:
        return self._trainer is not None

    def act_sample(self, image, instruction, proprio, wrist_image=None,
                   sigma: float = 0.0, seed: int | None = None) -> dict:
        if self._trainer is None:
            raise NotImplementedError("build the solver with trainable=True for online RL")
        wrist_image = image if wrist_image is None else wrist_image
        # Run the greedy path first: the normalized chunk is *captured* from inside the
        # upstream forward rather than recomputed, so sigma=0 sampling is bit-identical
        # to evaluation.
        mean_chunk = self.act(image, instruction, proprio, wrist_image)
        normalized = self._trainer.take_normalized()
        if normalized is None:
            raise RuntimeError("no normalized chunk captured (action head bypassed?)")
        rng = np.random.default_rng(int(seed)) if seed is not None else self._rng
        chunk, target = self._trainer.perturb_and_process(normalized, float(sigma), rng)
        sample_id = self._trainer.cache_sample(image, wrist_image, instruction, proprio)
        return {
            "action_chunk": chunk[: self.chunk_H],
            "normalized_chunk": target[: self.chunk_H],
            "mean_chunk": normalized[: self.chunk_H],
            "sample_id": sample_id,
            "greedy_chunk": mean_chunk,
        }

    def awr_update(self, items: list[dict], beta_l: float = 1.0, lr: float | None = None) -> dict:
        if self._trainer is None:
            raise NotImplementedError("build the solver with trainable=True for online RL")
        return self._trainer.awr_update(items, beta_l=beta_l, lr=lr)

    def clear_samples(self) -> dict:
        if self._trainer is None:
            return {"cleared": 0}
        return {"cleared": self._trainer.clear_samples()}

    def save_parameters(self, path: str, with_optim: bool = True) -> dict:
        if self._trainer is None:
            raise NotImplementedError("nothing to save: solver is not trainable")
        return self._trainer.save(path, with_optim=with_optim)

    def load_parameters(self, path: str) -> dict:
        if self._trainer is None:
            raise NotImplementedError("nothing to load: solver is not trainable")
        return self._trainer.load(path)

    def ping(self) -> dict:
        info = {
            "ok": True,
            "solver": "vla_adapter",
            "chunk_H": self.chunk_H,
            "action_dim": self.action_dim,
            "trainable": self.supports_training,
            "checkpoint": self.checkpoint,
        }
        if self._trainer is not None:
            info.update(self._trainer.info())
        return info
