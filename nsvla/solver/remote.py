"""Out-of-process action solver: a ZMQ client speaking to ``nsvla.solver.server``.

Two reasons a solver may need its own process. First, dependency isolation: the
reference VLA-Adapter checkpoint requires an older ``transformers`` than the encoder
side of this repository, so the two cannot share an interpreter. Second, placement:
several rollout workers on different GPUs can share one solver process, which keeps a
single owner for the low-level parameters theta during online RL.

``RemoteSolver`` implements the full ``ActionSolver`` surface, including the Stage-II
hooks, by forwarding each call over the wire. The wire contract is defined once, in
``nsvla.solver.server``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from nsvla.solver.base import ActionSolver, register_solver


@register_solver("remote")
class RemoteSolver(ActionSolver):
    """ZMQ REQ client. The socket is created lazily, so importing needs no server."""

    def __init__(
        self,
        cfg: Any = None,
        host: str = "127.0.0.1",
        port: int = 5678,
        timeout_ms: int = 60000,
        chunk_H: int = 8,
        action_dim: int = 7,
        max_retries: int = 2,
    ):
        if cfg is not None:
            host = getattr(cfg, "host", host)
            port = getattr(cfg, "port", port)
            timeout_ms = getattr(cfg, "request_timeout_ms", timeout_ms)
            chunk_H = getattr(cfg, "chunk_H", chunk_H)
            action_dim = getattr(cfg, "action_dim", action_dim)
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.chunk_H = chunk_H
        self.action_dim = action_dim
        self.max_retries = max_retries
        self._ctx = None
        self._sock = None
        self._trainable: bool | None = None

    # ---- transport ----------------------------------------------------- #
    def _ensure_socket(self) -> None:
        if self._sock is not None:
            return
        import zmq

        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(f"tcp://{self.host}:{self.port}")

    def _reset_socket(self) -> None:
        # A REQ socket is unusable after a timeout (its send/recv state is broken).
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None

    def _request(self, payload: dict) -> dict:
        import json_numpy
        import zmq

        blob = json_numpy.dumps(payload).encode("utf-8")
        last_err: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                self._ensure_socket()
                self._sock.send(blob)
                return json_numpy.loads(self._sock.recv().decode("utf-8"))
            except zmq.error.Again as e:
                last_err = e
                self._reset_socket()
        raise TimeoutError(
            f"solver server tcp://{self.host}:{self.port} timed out after "
            f"{self.max_retries + 1} attempts ({self.timeout_ms}ms each): {last_err!r}"
        )

    def _checked(self, payload: dict) -> dict:
        reply = self._request(payload)
        if not reply.get("ok", False):
            raise RuntimeError(
                f"solver server cmd {payload.get('cmd')!r} failed: {reply.get('error')}"
            )
        return reply

    @staticmethod
    def _obs(image, instruction, proprio, wrist_image) -> dict:
        return {
            "full_image": np.ascontiguousarray(image),
            # A solver consuming two views gets two; passing the primary view twice is
            # a degenerate fallback for callers that have no wrist camera.
            "wrist_image": np.ascontiguousarray(image if wrist_image is None else wrist_image),
            "instruction": instruction,
            "proprio": np.asarray(proprio, dtype=np.float32),
        }

    # ---- ActionSolver -------------------------------------------------- #
    def act(self, image, instruction, proprio, wrist_image=None) -> np.ndarray:
        reply = self._request(self._obs(image, instruction, proprio, wrist_image))
        if isinstance(reply, dict) and reply.get("error"):
            raise RuntimeError(f"solver server error: {reply['error']}")
        return np.asarray(reply["action_chunk"], dtype=np.float32)

    @property
    def supports_training(self) -> bool:
        if self._trainable is None:
            self._trainable = bool(self.ping().get("trainable", False))
        return self._trainable

    def act_sample(self, image, instruction, proprio, wrist_image=None,
                   sigma: float = 0.0, seed: int | None = None) -> dict:
        req = self._obs(image, instruction, proprio, wrist_image)
        req.update({"sample": True, "sigma": float(sigma), "seed": seed})
        reply = self._request(req)
        if isinstance(reply, dict) and reply.get("error"):
            raise RuntimeError(f"solver server error: {reply['error']}")
        return {
            "action_chunk": np.asarray(reply["action_chunk"], dtype=np.float32),
            "normalized_chunk": np.asarray(reply["normalized_chunk"], dtype=np.float32),
            "mean_chunk": np.asarray(reply["mean_chunk"], dtype=np.float32),
            "sample_id": reply["sample_id"],
        }

    def awr_update(self, items: list[dict], beta_l: float = 1.0, lr: float | None = None) -> dict:
        return self._checked({"cmd": "awr_update", "items": items, "beta_l": beta_l, "lr": lr})

    def clear_samples(self) -> dict:
        return self._checked({"cmd": "clear_samples"})

    def save_parameters(self, path: str, with_optim: bool = True) -> dict:
        # with_optim=False drops the optimizer state; the tagged best/final snapshots
        # are never resumed from, so they do not need it.
        return self._checked({"cmd": "save", "path": str(path), "with_optim": bool(with_optim)})

    def load_parameters(self, path: str) -> dict:
        return self._checked({"cmd": "load", "path": str(path)})

    def ping(self) -> dict:
        return self._request({"cmd": "ping"})

    def close(self) -> None:
        self._reset_socket()
