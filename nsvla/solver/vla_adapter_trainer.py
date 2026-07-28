"""Stage-II trainer for the low-level factor theta of the VLA-Adapter solver.

Instantiated by ``VLAAdapterSolver`` when it is built with ``trainable=True``. It
lives next to the solver because everything here touches VLA-Adapter internals; the
encoder side only ever sees the ``ActionSolver`` surface.

WHAT IS TRAINABLE (mechanism boundary, App. C "What is updated online"):

    theta = LoRA(r) adapters + action_head + proprio_projector

and nothing else - the VLM encoder, the plan generator and the pointer's inputs stay
frozen. The LoRA is attached on top of the Stage-I BC checkpoint (already merged),
with ``init_lora_weights="gaussian"`` so that B = 0 at step 0: the policy at iteration
0 is bit-identical to the BC policy, and disabling the adapter recovers the frozen BC
reference exactly at any later time.

EXPLORATION (Alg.1 line 18, "sample an action chunk A_t ~ pi_theta_old"). The action
head is a deterministic L1 regressor, so the stochastic policy is
``A_t = mu_theta(o_t) + eps``, ``eps ~ N(0, sigma^2 I)``, with the noise added in the
**normalized** action space (the [-1, 1] scale the head regresses in) and clipped
there before un-normalization. AWR needs no density - its low-level loss is an
advantage-weighted L1 regression onto the executed chunk - so a deterministic head
plus Gaussian dithering is a complete instantiation of Eq. 9's low-level term.

FIDELITY. The rollout path is the upstream one verbatim; the normalized chunk is
obtained by *capturing* the action head's output inside that call (``_CaptureHead``),
never by re-deriving it, so a sampled rollout with sigma=0 is bit-identical to an
evaluation rollout. Only the gradient path re-runs the forward, because the upstream
one is inside ``torch.inference_mode()`` and its outputs cannot carry grad; that
re-run mirrors ``modeling_prismatic.predict_action`` line for line.
"""
from __future__ import annotations

import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# normalized-chunk capture (rollout path, zero divergence from official)
# --------------------------------------------------------------------------- #
class _CaptureHead:
    """Wrap ``action_head.predict_action`` so the NORMALIZED chunk can be read back.

    The upstream inference path throws the normalized tensor away: it un-normalizes
    and returns numpy. AWR regresses in normalized space and needs it, but forking the
    inference path to obtain it would break bit-equality with evaluation, so this
    installs a transparent wrapper that passes the return value through untouched.
    """

    def __init__(self, action_head):
        self.action_head = action_head
        self._orig = action_head.predict_action
        self.last: torch.Tensor | None = None
        action_head.predict_action = self._wrapped

    def _wrapped(self, *a, **k):
        out = self._orig(*a, **k)
        self.last = out.detach().float().cpu()
        return out

    def restore(self) -> None:
        self.action_head.predict_action = self._orig


# --------------------------------------------------------------------------- #
# trainer
# --------------------------------------------------------------------------- #
class VLAAdapterTrainer:
    """Owns theta, the AWR optimizer, and the rollout-sample cache."""

    def __init__(
        self,
        cfg,
        vla,
        action_head,
        proprio_projector,
        processor,
        lora_rank: int = 64,
        lora_target: str = "all-linear",
        lr: float = 1e-5,
        grad_clip: float = 1.0,
        sample_cache_cap: int = 4096,
        freeze: bool = False,
    ):
        from peft import LoraConfig, get_peft_model

        self.cfg = cfg
        self.processor = processor
        self.action_head = action_head
        self.proprio_projector = proprio_projector
        self.grad_clip = grad_clip
        self.lr = lr

        # ---- attach a zero-initialized LoRA on top of the BC checkpoint ----
        target = "all-linear" if lora_target == "all-linear" else _llm_linear_names(vla)
        lora_cfg = LoraConfig(
            r=lora_rank,
            lora_alpha=2 * lora_rank,
            lora_dropout=0.0,
            target_modules=target,
            init_lora_weights="gaussian",   # A ~ N(0, .), B = 0  =>  delta(0) = 0
        )
        self.vla = get_peft_model(vla, lora_cfg)
        self.base = vla                      # LoRA layers are injected IN PLACE here
        self.lora_target = lora_target

        for p in self.action_head.parameters():
            p.requires_grad_(True)
        if self.proprio_projector is not None:
            for p in self.proprio_projector.parameters():
                p.requires_grad_(True)

        lora_params = [p for p in self.vla.parameters() if p.requires_grad]
        head_params = [p for p in self.action_head.parameters() if p.requires_grad]
        proprio_params = ([p for p in self.proprio_projector.parameters() if p.requires_grad]
                          if self.proprio_projector is not None else [])
        params = lora_params + head_params + proprio_params
        self.trainable = params
        self.n_params = {
            "lora": int(sum(p.numel() for p in lora_params)),
            "action_head": int(sum(p.numel() for p in head_params)),
            "proprio_projector": int(sum(p.numel() for p in proprio_params)),
        }
        self.n_trainable = int(sum(self.n_params.values()))
        # ---- theta-frozen ablation ------------------------------------------------
        # Isolates "did RL change the pointer's timing?" from any change in the
        # low-level factor. With ``freeze=True`` everything that shapes a rollout is
        # kept -- the LoRA is still attached (B=0, so the policy equals the BC
        # checkpoint), the normalized-chunk capture still runs and the sigma
        # exploration noise of Alg.1 line 18 is still applied -- but theta has no
        # optimizer and no gradient path, so only phi moves. Both the optimizer and
        # requires_grad are cleared, leaving no route by which a stray backward could
        # touch theta.
        self._theta_groups = {
            "lora": lora_params,
            "action_head": head_params,
            "proprio_projector": proprio_params,
        }
        self.frozen = bool(freeze)
        if self.frozen:
            for p in params:
                p.requires_grad_(False)
            self.opt = None
        else:
            # foreach=False keeps the optimizer step's transient allocation to one
            # tensor at a time, which is what makes it fit on a shared card.
            self.opt = torch.optim.AdamW(params, lr=lr, foreach=False)
        self.step_count = 0

        self.capture = _CaptureHead(action_head)
        self._samples: OrderedDict[str, dict] = OrderedDict()
        self.sample_cache_cap = sample_cache_cap

    # ------------------------------------------------------------------ #
    # rollout side
    # ------------------------------------------------------------------ #
    def take_normalized(self) -> np.ndarray | None:
        """Normalized chunk captured by the most recent official forward, (chunk, dim)."""
        from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

        if self.capture.last is None:
            return None
        return self.capture.last.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM).numpy()

    def perturb_and_process(
        self, normalized: np.ndarray, sigma: float, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        """Alg.1 line 18: At = mu + eps, eps~N(0,sigma^2); -> (executed chunk, target).

        The perturbation happens in normalized space and is clipped to the head's own
        [-1, 1] range there; the returned executed chunk goes through the upstream
        un-normalization and ``process_action``, so the environment sees exactly what
        upstream would send for that (noised) normalized action.
        """
        from experiments.robot.libero.run_libero_eval import process_action

        noised = normalized.astype(np.float32)
        if sigma > 0:
            noised = noised + rng.normal(0.0, sigma, size=noised.shape).astype(np.float32)
            noised = np.clip(noised, -1.0, 1.0)
        unnorm = self.base._unnormalize_actions(noised, self.cfg.unnorm_key)
        processed = np.stack(
            [process_action(np.asarray(a), self.cfg.model_family) for a in unnorm], axis=0
        ).astype(np.float32)
        return processed, noised

    def cache_sample(self, full_image, wrist_image, instruction, proprio) -> str:
        """Stash the observation of one decision step; return its handle.

        The AWR update needs a grad-enabled re-forward of the SAME observation. Sending
        the two renders back over the wire per chunk would dominate the wire cost, so
        the solver keeps them (uint8, host RAM) and the caller round-trips only a handle
        plus a float target and advantage.
        """
        sid = uuid.uuid4().hex[:16]
        self._samples[sid] = {
            "full_image": np.ascontiguousarray(full_image, dtype=np.uint8),
            "wrist_image": np.ascontiguousarray(wrist_image, dtype=np.uint8),
            "instruction": str(instruction),
            "proprio": np.asarray(proprio, dtype=np.float64),
        }
        while len(self._samples) > self.sample_cache_cap:
            self._samples.popitem(last=False)
        return sid

    def clear_samples(self) -> int:
        n = len(self._samples)
        self._samples.clear()
        return n

    # ------------------------------------------------------------------ #
    # freeze audit
    # ------------------------------------------------------------------ #
    def theta_fingerprint(self) -> dict:
        """sum(p^2) per theta group — a cheap, exact "did theta move?" witness.

        Reported inside ``info()``, hence by every ``ping``, so "theta was frozen for
        the whole run" is checkable from the wire: ping before the first iteration and
        after the last, and compare. A single optimizer step on any of the three groups
        moves the corresponding float far above print precision.
        """
        out: dict[str, float] = {}
        with torch.no_grad():
            for name, ps in self._theta_groups.items():
                out[name] = float(sum(float(p.detach().float().pow(2).sum()) for p in ps))
        return out

    # ------------------------------------------------------------------ #
    # gradient side
    # ------------------------------------------------------------------ #
    def _build_inputs(self, full_image, wrist_image, instruction, proprio):
        """Upstream preprocessing (mirrors ``openvla_utils.get_vla_action``)."""
        from experiments.robot.openvla_utils import (
            DEVICE, normalize_proprio, prepare_images_for_vla,
        )

        images = prepare_images_for_vla(
            [np.ascontiguousarray(full_image), np.ascontiguousarray(wrist_image)], self.cfg
        )
        primary = images.pop(0)
        if not self.cfg.use_minivlm:
            prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        else:
            prompt = (
                "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a "
                "helpful assistant.<|im_end|>\n<|im_start|>user\nWhat action should the "
                f"robot take to {instruction.lower()}?<|im_end|>\n<|im_start|>assistant\n"
            )
        inputs = self.processor(prompt, primary).to(DEVICE, dtype=torch.bfloat16)
        if images:
            wrist_inputs = [
                self.processor(prompt, im).to(DEVICE, dtype=torch.bfloat16) for im in images
            ]
            inputs["pixel_values"] = torch.cat(
                [inputs["pixel_values"]] + [w["pixel_values"] for w in wrist_inputs], dim=1
            )
        pn = None
        if self.cfg.use_proprio:
            pn = normalize_proprio(
                np.asarray(proprio, dtype=np.float64),
                self.base.norm_stats[self.cfg.unnorm_key]["proprio"],
            )
        return inputs, pn

    def forward_normalized_grad(self, sample: dict) -> torch.Tensor:
        """Grad-enabled (chunk, dim) normalized action — mirrors ``predict_action``.

        Line-for-line the same computation as
        ``modeling_prismatic.OpenVLAForActionPrediction.predict_action`` up to and
        including ``action_head.predict_action``; the only differences are that it runs
        outside ``inference_mode`` and does not detach, which is what a gradient step
        needs. LoRA is injected in place, so calling the submodules directly still
        routes through the trainable adapters.
        """
        from prismatic.vla.constants import (
            ACTION_DIM, IGNORE_INDEX, NUM_ACTIONS_CHUNK, NUM_TOKENS,
        )

        vla = self.base
        inputs, proprio = self._build_inputs(
            sample["full_image"], sample["wrist_image"],
            sample["instruction"], sample["proprio"],
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        pixel_values = inputs["pixel_values"]

        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        num_prompt_tokens = input_ids.shape[-1] - 1
        input_ids, attention_mask = vla._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = vla._prepare_labels_for_action_prediction(labels, input_ids)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            input_embeddings = vla.get_input_embeddings()(input_ids)
            all_actions_mask = vla._process_action_masks(labels)
            language_embeddings = input_embeddings[~all_actions_mask].reshape(
                input_embeddings.shape[0], -1, input_embeddings.shape[2]
            )
            projected_patch_embeddings = vla._process_vision_features(
                pixel_values, language_embeddings, False
            )
            num_patches = (
                vla.vision_backbone.get_num_patches()
                * vla.vision_backbone.get_num_images_in_input()
            )

            action_queries = vla.action_queries.weight
            aq = action_queries.view(1, action_queries.shape[0], action_queries.shape[1]).repeat(
                input_embeddings.shape[0], 1, 1
            )
            input_embeddings = vla._replace_input_embeddings(
                input_embeddings.clone(), all_actions_mask, aq
            )
            mm_emb, mm_mask = vla._build_multimodal_attention(
                input_embeddings, projected_patch_embeddings, attention_mask
            )
            out = vla.language_model(
                input_ids=None,
                attention_mask=mm_mask,
                position_ids=None,
                past_key_values=None,
                inputs_embeds=mm_emb,
                labels=None,
                use_cache=False,
                output_attentions=False,
                output_hidden_states=True,
                return_dict=True,
            )

            multi = []
            for item in out.hidden_states:
                b = item.shape[0]
                ah = item[
                    :, num_patches + num_prompt_tokens : num_patches + num_prompt_tokens + NUM_TOKENS, :
                ].reshape(b, 1, NUM_TOKENS, -1).to(torch.bfloat16)
                tl = item[:, :num_patches].reshape(b, 1, num_patches, -1)
                multi.append(torch.cat((tl, ah), 2))
            multi = torch.cat(multi, dim=1)

            proprio_t = None
            if proprio is not None:
                proprio_t = torch.as_tensor(
                    proprio, device=multi.device, dtype=multi.dtype
                )
            normalized = self.action_head.predict_action(
                multi, proprio=proprio_t, proprio_projector=self.proprio_projector
            )
        return normalized.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

    def awr_update(
        self,
        items: list[dict],
        beta_l: float = 1.0,
        lr: float | None = None,
    ) -> dict:
        """One AWR step on theta (Eq. 9 low level).

            w_t = softmax_t(A^l_t / beta_l)          (detached, normalized over the batch)
            L^l = sum_t w_t * || pi^l_theta(e_t) - A_t ||_1

        The forward is batch-1 (``modeling_prismatic`` assumes it), so the batch
        gradient is accumulated one chunk at a time and a single optimizer step is
        taken: numerically identical to a full-batch step, with a bounded memory peak.
        """
        usable = [it for it in items if it.get("sample_id") in self._samples]
        if not usable:
            return {"n": 0, "n_missing": len(items), "loss_low": 0.0}

        if self.frozen:
            # theta-frozen ablation: the caller's low-level update still runs end to
            # end (same advantages, same subsampling, same trace bookkeeping) and still
            # ships the batch, so the only thing removed is the gradient step itself.
            # Reporting n / n_missing keeps that visible in the metrics stream.
            return {
                "n": len(usable), "n_missing": len(items) - len(usable),
                "loss_low": 0.0, "frozen_theta": True, "step_count": self.step_count,
            }

        adv = torch.tensor([float(it["advantage"]) for it in usable], dtype=torch.float32)
        w = torch.softmax(adv / max(float(beta_l), 1e-6), dim=0)     # detached by construction

        if lr is not None and lr != self.lr:
            self.lr = float(lr)
            for g in self.opt.param_groups:
                g["lr"] = self.lr

        self.opt.zero_grad(set_to_none=True)
        total = 0.0
        l1_sum = 0.0
        for j, it in enumerate(usable):
            sample = self._samples[it["sample_id"]]
            pred = self.forward_normalized_grad(sample)
            target = torch.as_tensor(
                np.asarray(it["target"], dtype=np.float32), device=pred.device, dtype=pred.dtype
            )
            l1 = (pred - target).abs().sum()
            loss_j = float(w[j]) * l1
            loss_j.backward()
            total += float(loss_j.detach())
            l1_sum += float(l1.detach())

        gnorm = torch.nn.utils.clip_grad_norm_(self.trainable, self.grad_clip)
        self.opt.step()
        self.step_count += 1
        return {
            "n": len(usable),
            "n_missing": len(items) - len(usable),
            "loss_low": total,
            "mean_l1": l1_sum / max(1, len(usable)),
            "w_max": float(w.max()),
            "grad_norm_low": float(gnorm),
            "lr_low": self.lr,
            "step_count": self.step_count,
        }

    # ------------------------------------------------------------------ #
    # checkpointing
    # ------------------------------------------------------------------ #
    def save(self, path: str, with_optim: bool = True) -> dict:
        """Persist theta. ``with_optim`` controls the AdamW state (~0.8 GB by itself).

        Only the resume snapshot needs the optimizer; the tagged best and final
        snapshots are weights-only, roughly a third of the size. Over a full run that
        is the difference between fitting on a small partition and not.
        """
        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        if self.frozen:
            # theta never moved, so the snapshot IS the Stage-I checkpoint. Writing
            # provably unchanged weights every ckpt_every iterations would only consume
            # disk, so leave a fingerprinted marker instead.
            import json as _json

            (out / "frozen_theta.json").write_text(_json.dumps({
                "frozen_theta": True,
                "source_checkpoint": str(self.cfg.pretrained_checkpoint),
                "theta_fingerprint": self.theta_fingerprint(),
                "step_count": self.step_count,
            }, indent=2))
            return {"saved": str(out), "step_count": self.step_count,
                    "with_optim": False, "frozen_theta": True, "weights_written": False}
        self.vla.save_pretrained(str(out / "lora_adapter"))
        torch.save(self.action_head.state_dict(), out / "action_head.pt")
        if self.proprio_projector is not None:
            torch.save(self.proprio_projector.state_dict(), out / "proprio_projector.pt")
        if with_optim:
            torch.save(
                {"opt": self.opt.state_dict(), "step_count": self.step_count, "lr": self.lr},
                out / "awr_optim.pt",
            )
        else:
            (out / "awr_optim.pt").unlink(missing_ok=True)
        return {"saved": str(out), "step_count": self.step_count, "with_optim": with_optim}

    def load(self, path: str) -> dict:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        src = Path(path)
        ad = src / "lora_adapter"
        sd_path = ad / "adapter_model.safetensors"
        if sd_path.exists():
            set_peft_model_state_dict(self.vla, load_file(str(sd_path)))
        elif (ad / "adapter_model.bin").exists():
            set_peft_model_state_dict(self.vla, torch.load(ad / "adapter_model.bin", map_location="cpu"))
        if (src / "action_head.pt").exists():
            self.action_head.load_state_dict(torch.load(src / "action_head.pt", map_location="cpu"))
        if self.proprio_projector is not None and (src / "proprio_projector.pt").exists():
            self.proprio_projector.load_state_dict(
                torch.load(src / "proprio_projector.pt", map_location="cpu")
            )
        if (src / "awr_optim.pt").exists() and self.opt is not None:
            st = torch.load(src / "awr_optim.pt", map_location="cpu")
            self.opt.load_state_dict(st["opt"])
            self.step_count = int(st.get("step_count", 0))
        return {"loaded": str(src), "step_count": self.step_count}

    def info(self) -> dict:
        return {
            # "trainable" advertises the Stage-II surface (sample mode plus the
            # awr/save/load commands), which is what the trainer's start-up guard
            # checks; whether theta actually moves is the separate ``frozen_theta`` flag.
            "trainable": True,
            "frozen_theta": self.frozen,
            "n_trainable_params": 0 if self.frozen else self.n_trainable,
            "n_params_breakdown": self.n_params,
            "lora_target": self.lora_target,
            "lr_low": self.lr,
            "step_count": self.step_count,
            "n_cached_samples": len(self._samples),
            "theta_fingerprint": self.theta_fingerprint(),
        }


def _llm_linear_names(vla) -> list[str]:
    """Linear-module suffixes inside the language model only (memory-lean LoRA variant)."""
    names = set()
    for n, m in vla.named_modules():
        if isinstance(m, torch.nn.Linear) and n.startswith("language_model"):
            names.add(n.split(".")[-1])
    names.discard("lm_head")
    return sorted(names)
