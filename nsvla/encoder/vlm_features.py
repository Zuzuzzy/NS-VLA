"""Frozen VLM context features  psi_t  and pooled  psi_bar_t  (NS-VLA §4.1, Eq. 5).

A frozen vision-language model maps ``(o_t, x)`` -> token features
``psi_t in R^{N x d_psi}``, mean-pooled into the context vector
``psi_bar_t in R^{d_psi}`` consumed by the monotone pointer classifier ``g_phi``.

Wiring: ``AutoProcessor`` + ``AutoModelForImageTextToText`` -> chat template over
(image, text) -> forward with ``output_hidden_states=True`` -> ``hidden_states[-1]``
as the token features psi_t, attention-masked mean pool -> psi_bar_t.

The encoder is frozen, so features are pre-extracted once and cached on disk: one
``psi_bar`` per sampled frame (every H-chunk boundary), plus optionally the visual
token features the App. F sparsifier consumes. Cache location and model directory
come from ``nsvla.utils.paths`` and are environment-overridable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from nsvla.utils import paths

DEFAULT_MODEL_DIR = paths.vlm_model_dir()
FEATURE_ROOT = paths.feature_root()


def chunk_boundary_indices(n_frames: int, chunk_H: int) -> list[int]:
    """Frame indices sampled at every H-chunk boundary: 0, H, 2H, ... (< n_frames).

    One VLM feature per chunk boundary. The final frame is always included so the
    segment-end window is represented.
    """
    if n_frames <= 0:
        return []
    idx = list(range(0, n_frames, max(1, chunk_H)))
    if idx[-1] != n_frames - 1:
        idx.append(n_frames - 1)
    return idx


def _to_pil(image: Any):
    from PIL import Image

    if isinstance(image, Image.Image):
        return image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


class VLMFeatureEncoder:
    """Frozen VLM feature extractor (lazy GPU load)."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_DIR,
        device: str = "cuda",
        dtype: str = "bfloat16",
        image_size: int = 224,
    ):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype
        self.image_size = image_size
        self._model = None
        self._processor = None
        self._image_token_id: int | None = None

    @property
    def feature_dim(self) -> int:
        return 2048  # text hidden size of the reference 2B encoder

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        import os

        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        # A local directory that does not exist is otherwise reported as an invalid
        # Hub repo id, which says nothing about which knob is wrong.
        if os.sep in str(self.model_id) and not os.path.isdir(self.model_id):
            raise FileNotFoundError(
                f"frozen VLM encoder weights not found at {self.model_id!r}. Download the "
                f"encoder and point NSVLA_VLM_MODEL at it, or pass a Hub repo id. Policies "
                f"that do not use the pointer (--policy solver_bare) need no encoder."
            )
        torch_dtype = getattr(torch, self.dtype)
        self._processor = AutoProcessor.from_pretrained(self.model_id)
        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            dtype=torch_dtype,
            device_map=self.device,
            attn_implementation="sdpa",
        )
        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        cfg = self._model.config
        self._image_token_id = getattr(cfg, "image_token_id", None)

    # ------------------------------------------------------------------ #
    # core batched forward
    # ------------------------------------------------------------------ #
    def _build_batch(self, images: Sequence[Any], instruction: str):
        """One image per sample, shared instruction -> processor tensors on device."""
        proc = self._processor
        pil_images = [_to_pil(im) for im in images]
        texts = []
        for _ in pil_images:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]
            texts.append(
                proc.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            )
        inputs = proc(
            text=texts,
            images=pil_images,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self._model.device) for k, v in inputs.items()}

    def encode_frames(
        self,
        images: Sequence[Any],
        instruction: str,
        return_visual_tokens: bool = False,
        topk_visual: int | None = None,
        return_shaping: bool = False,
    ) -> dict[str, np.ndarray]:
        """Batched extraction for frames sharing one instruction.

        Returns a dict with:
          * ``psi_bar``       : (B, d) attention-masked mean-pooled last hidden state.
          * ``visual_tokens`` : (B, Kv, d) per-frame visual (image) token features,
            only if ``return_visual_tokens``. If ``topk_visual`` is set, each frame is
            truncated/padded to that many visual tokens (norm-ranked) to bound disk.
          * ``ell``           : (B, d) the **shaping latent** ``ell_t = E_w(o_t)``
            (Alg.1 line 16) - the mean over this frame's IMAGE tokens only, i.e. the
            visual pooling of the same frozen forward. Sharing one forward between the
            pointer feature and the shaping encoder is what keeps a rollout worker
            inside a few GB of VRAM; ``E_w`` is frozen either way, so the prototype
            potential is unaffected.
        """
        if self._model is None:
            self.load()
        import torch

        inputs = self._build_batch(images, instruction)
        with torch.no_grad():
            out = self._model(**inputs, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states[-1]                      # (B, T, d) bf16
        mask = inputs["attention_mask"].unsqueeze(-1).to(hs.dtype)  # (B, T, 1)
        summed = (hs * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        psi_bar = (summed / counts).float().cpu().numpy()

        result: dict[str, np.ndarray] = {"psi_bar": psi_bar}

        if return_visual_tokens:
            vis = self._extract_visual_tokens(hs, inputs["input_ids"], topk_visual)
            result["visual_tokens"] = vis
        if return_shaping:
            result["ell"] = self._pool_visual(hs, inputs["input_ids"])
        return result

    def _pool_visual(self, hs, input_ids) -> np.ndarray:
        """Mean over image-token hidden states -> the frozen shaping latent (B, d)."""
        import torch

        B, T, d = hs.shape
        img_id = self._image_token_id
        rows = []
        for b in range(B):
            if img_id is not None:
                sel = input_ids[b] == img_id
                if not bool(sel.any()):
                    sel = torch.ones(T, dtype=torch.bool, device=hs.device)
            else:
                sel = torch.ones(T, dtype=torch.bool, device=hs.device)
            rows.append(hs[b][sel].float().mean(dim=0))
        return torch.stack(rows, dim=0).cpu().numpy()

    def _extract_visual_tokens(self, hs, input_ids, topk_visual):
        """Per-frame image-token hidden states (B, Kv, d), norm-ranked & padded."""
        import torch

        B, T, d = hs.shape
        img_id = self._image_token_id
        rows = []
        for b in range(B):
            if img_id is not None:
                sel = input_ids[b] == img_id
            else:
                sel = torch.ones(T, dtype=torch.bool, device=hs.device)
            tok = hs[b][sel]                              # (n_b, d)
            if tok.shape[0] == 0:
                tok = hs[b]
            if topk_visual is not None and tok.shape[0] > topk_visual:
                order = tok.float().norm(dim=-1).argsort(descending=True)[:topk_visual]
                tok = tok[order.sort().values]
            rows.append(tok.float().cpu().numpy())
        kv = max(r.shape[0] for r in rows)
        out = np.zeros((B, kv, d), dtype=np.float32)
        for b, r in enumerate(rows):
            out[b, : r.shape[0]] = r
        return out

    # ------------------------------------------------------------------ #
    # single-frame convenience (Eq. 5)
    # ------------------------------------------------------------------ #
    def context(self, image: Any, instruction: str) -> np.ndarray:
        """(image224, instruction) -> mean-pooled psi_bar in R^{d} (Eq. 5)."""
        return self.encode_frames([image], instruction)["psi_bar"][0]

    def context_and_shaping(self, image: Any, instruction: str) -> tuple[np.ndarray, np.ndarray]:
        """(image, instruction) -> (psi_bar for Eq.5, ell_t = E_w(o_t) for Alg.1 line 16).

        ONE frozen forward feeds both consumers (see ``encode_frames``).
        """
        res = self.encode_frames([image], instruction, return_shaping=True)
        return res["psi_bar"][0], res["ell"][0]

    def tokens(self, image: Any, instruction: str) -> np.ndarray:
        """(image224, instruction) -> visual token features psi in R^{Kv x d}."""
        return self.encode_frames(
            [image], instruction, return_visual_tokens=True
        )["visual_tokens"][0]

    # ------------------------------------------------------------------ #
    # disk cache
    # ------------------------------------------------------------------ #
    def cache_demo(
        self,
        images: Sequence[Any],
        instruction: str,
        out_path: str | Path,
        frame_indices: Sequence[int] | None = None,
        chunk_H: int = 8,
        batch_size: int = 16,
        save_visual_tokens: bool = False,
        topk_visual: int | None = 32,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Extract & cache psi_bar (and optional visual tokens) for one demo.

        ``images`` is the full per-frame RGB stack; frames are subsampled at every
        ``chunk_H`` boundary (unless ``frame_indices`` given), embedded in mini-batches,
        and written to ``out_path`` via ``np.savez_compressed``. Returns a manifest dict.
        """
        n = len(images)
        if frame_indices is None:
            frame_indices = chunk_boundary_indices(n, chunk_H)
        frame_indices = list(frame_indices)
        sel = [images[i] for i in frame_indices]

        psi_chunks: list[np.ndarray] = []
        vis_chunks: list[np.ndarray] = []
        for s in range(0, len(sel), batch_size):
            batch = sel[s : s + batch_size]
            res = self.encode_frames(
                batch,
                instruction,
                return_visual_tokens=save_visual_tokens,
                topk_visual=topk_visual,
            )
            psi_chunks.append(res["psi_bar"])
            if save_visual_tokens:
                vis_chunks.append(res["visual_tokens"])
        psi_bar = np.concatenate(psi_chunks, axis=0)

        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "psi_bar": psi_bar,
            "frame_indices": np.asarray(frame_indices, dtype=np.int64),
            "n_frames": np.int64(n),
            "chunk_H": np.int64(chunk_H),
            "instruction": np.array(instruction),
        }
        if save_visual_tokens:
            # ragged across batches only if kv differs; pad to global max.
            kv = max(v.shape[1] for v in vis_chunks)
            d = vis_chunks[0].shape[2]
            vt = np.zeros((psi_bar.shape[0], kv, d), dtype=np.float32)
            off = 0
            for v in vis_chunks:
                vt[off : off + v.shape[0], : v.shape[1]] = v
                off += v.shape[0]
            payload["visual_tokens"] = vt
        if extra:
            payload.update(extra)
        np.savez_compressed(out_path, **payload)
        return {
            "out_path": str(out_path),
            "n_sampled": int(psi_bar.shape[0]),
            "frame_indices": frame_indices,
            "d": int(psi_bar.shape[1]),
        }


def feature_cache_path(suite: str, task: str, demo: str, root: str = FEATURE_ROOT) -> Path:
    """Canonical cache path ``<feature root>/<suite>/<task>/<demo>.npz``."""
    return Path(root) / suite / task / f"{demo}.npz"


def load_cached_features(path: str | Path) -> dict[str, Any]:
    """Load a demo feature cache into a plain dict (psi_bar, frame_indices, ...)."""
    with np.load(path, allow_pickle=True) as z:
        out = {k: z[k] for k in z.files}
    if "instruction" in out:
        out["instruction"] = str(out["instruction"])
    return out
