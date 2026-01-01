"""The action-solver interface (NS-VLA §4.2, Eq. 7).

The neuro-symbolic encoder and the online RL stage never touch a policy network
directly: they talk to an ``ActionSolver``, which maps one observation plus the
primitive-conditioned instruction to an H-step action chunk. Any VLA or policy model
can be plugged in by implementing this interface and registering it; the VLA-Adapter
solver in ``nsvla.solver.vla_adapter`` is the reference implementation used in the
paper, and ``nsvla.solver.remote`` is a transport wrapper that lets a solver live in a
separate process (or on another machine) behind ZMQ.

Required surface (evaluation and Stage-I):

    act(image, instruction, proprio, wrist_image=None) -> (H, action_dim) float32

Optional surface (Stage-II online RL). A solver that does not implement these can
still be evaluated and can still serve the supervised stage; ``supports_training``
reports which half is available.

    act_sample(...)   -> {action_chunk, normalized_chunk, mean_chunk, sample_id}
    awr_update(items) -> metrics          advantage-weighted regression step on theta
    save_parameters / load_parameters
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np


class ActionSolver(ABC):
    """Instruction + observation -> action chunk. The only policy contract NS-VLA needs."""

    #: chunk length H executed open-loop per decision step (Eq. 7)
    chunk_H: int = 8
    #: dimensionality of a single action
    action_dim: int = 7

    # ---- required ---------------------------------------------------- #
    @abstractmethod
    def act(
        self,
        image: np.ndarray,
        instruction: str,
        proprio: np.ndarray,
        wrist_image: np.ndarray | None = None,
    ) -> np.ndarray:
        """Greedy action chunk of shape ``(chunk_H, action_dim)``.

        ``instruction`` is where NS-VLA injects its conditioning: the solver receives
        the rendered sub-instruction ``x_tilde_t`` for the active primitive, not the
        raw task string.
        """

    # ---- optional (Stage-II) ------------------------------------------ #
    @property
    def supports_training(self) -> bool:
        """True when ``act_sample`` / ``awr_update`` are available."""
        return False

    def act_sample(
        self,
        image: np.ndarray,
        instruction: str,
        proprio: np.ndarray,
        wrist_image: np.ndarray | None = None,
        sigma: float = 0.0,
        seed: int | None = None,
    ) -> dict:
        """Alg.1 line 18: sample ``A_t ~ pi_theta_old``.

        Returns ``action_chunk`` (what the environment executes), ``normalized_chunk``
        (the same chunk in the head's own action space, which is the AWR regression
        target), ``mean_chunk`` (the unperturbed output) and ``sample_id`` (a handle
        the solver can use to replay this observation during ``awr_update``).
        """
        raise NotImplementedError(f"{type(self).__name__} does not support online RL")

    def awr_update(self, items: list[dict], beta_l: float = 1.0, lr: float | None = None) -> dict:
        """One advantage-weighted regression step on theta (Eq. 9, low level).

        ``items`` = ``[{sample_id, target (H, action_dim), advantage}]``.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support online RL")

    def clear_samples(self) -> dict:
        return {"cleared": 0}

    def save_parameters(self, path: str, with_optim: bool = True) -> dict:
        raise NotImplementedError(f"{type(self).__name__} has no trainable parameters")

    def load_parameters(self, path: str) -> dict:
        raise NotImplementedError(f"{type(self).__name__} has no trainable parameters")

    # ---- lifecycle ----------------------------------------------------- #
    def load(self) -> None:
        """Acquire whatever the solver needs before it can answer (weights, devices).

        Called once by ``nsvla.solver.server.serve`` before the socket is bound, so a
        misconfigured solver fails at startup instead of looking healthy and erroring on
        the first request. Solvers that need nothing may leave this as a no-op.
        """

    def ping(self) -> dict:
        """Health / capability report, also used as the wire handshake."""
        return {
            "ok": True,
            "solver": type(self).__name__,
            "chunk_H": self.chunk_H,
            "action_dim": self.action_dim,
            "trainable": self.supports_training,
        }

    def close(self) -> None:
        pass


# --------------------------------------------------------------------------- #
# registry
# --------------------------------------------------------------------------- #
SOLVER_REGISTRY: dict[str, Callable[..., ActionSolver]] = {}

# Solvers whose module pulls in dependencies this package does not require. They are
# imported on first use rather than at package import, so ``import nsvla`` never needs
# a third-party policy installed; ``make_solver`` triggers the import and turns a
# missing dependency into an actionable message instead of a bare ImportError.
LAZY_SOLVERS: dict[str, tuple[str, str]] = {
    "vla_adapter": (
        "nsvla.solver.vla_adapter",
        "the VLA-Adapter source tree (set NSVLA_SOLVER_ROOT or --source-root), plus "
        "its own environment: transformers==4.40, peft, safetensors",
    ),
}


def register_solver(name: str) -> Callable[[type], type]:
    """Class decorator adding a solver to the registry under ``name``."""

    def wrap(cls: type) -> type:
        SOLVER_REGISTRY[name] = cls
        return cls

    return wrap


def load_solver_module(name: str) -> bool:
    """Import a lazily registered solver. True if ``name`` is now in the registry."""
    if name in SOLVER_REGISTRY:
        return True
    if name not in LAZY_SOLVERS:
        return False
    module, requires = LAZY_SOLVERS[name]
    import importlib

    try:
        importlib.import_module(module)
    except ImportError as e:
        raise ImportError(
            f"solver {name!r} is available but its dependencies are not installed: "
            f"{e}. It requires {requires}."
        ) from e
    return name in SOLVER_REGISTRY


def make_solver(spec: Any, **overrides: Any) -> ActionSolver:
    """Build a solver from a name or from a config object.

    ``spec`` is either the registered name (``make_solver("fake")``, which builds the
    solver with its own defaults) or a ``SolverConfig``-like object carrying a ``name``
    field, whose remaining fields are handed to the solver class so a third-party solver
    only has to accept the subset it cares about.

    An object without a usable ``name`` is a ``TypeError``, never a default. Silently
    falling back would hand back a working-looking solver of the wrong kind, and the
    mistake would not surface until the first action request.
    """
    if isinstance(spec, str):
        name, cfg = spec, None
    else:
        name = getattr(spec, "name", None)
        if not isinstance(name, str) or not name:
            raise TypeError(
                f"make_solver() expects a solver name or a config with a 'name' field, "
                f"got {type(spec).__name__} with name={name!r}. Use "
                f"make_solver('fake') or make_solver(SolverConfig(name='fake'))."
            )
        cfg = spec

    if not load_solver_module(name):
        known = sorted(set(SOLVER_REGISTRY) | set(LAZY_SOLVERS))
        raise KeyError(
            f"unknown solver {name!r}; available: {known}. "
            f"Implement ActionSolver and decorate it with @register_solver(...)."
        )
    return SOLVER_REGISTRY[name](cfg, **overrides)
