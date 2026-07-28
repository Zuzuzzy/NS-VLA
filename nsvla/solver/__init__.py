"""Action solver: the interface between the symbolic layer and any policy model.

``ActionSolver`` (in ``base``) is the contract; ``remote`` runs a solver in another
process over ZMQ, ``server`` is the matching server side, ``vla_adapter`` is the
reference implementation and ``fake`` is a deterministic CPU stand-in.

Importing a solver module is what registers it. The dependency-free ones are imported
here; ``vla_adapter`` is listed in ``base.LAZY_SOLVERS`` and imported by ``make_solver``
on first use, so ``import nsvla`` never requires a third-party policy to be installed
and asking for a solver whose dependencies are missing reports what to install.
"""
from nsvla.solver.base import (  # noqa: F401
    LAZY_SOLVERS,
    SOLVER_REGISTRY,
    ActionSolver,
    load_solver_module,
    make_solver,
    register_solver,
)
from nsvla.solver.bridge import PrimitiveBridge, render_sub_instruction  # noqa: F401
from nsvla.solver.fake import FakeSolver  # noqa: F401
from nsvla.solver.remote import RemoteSolver  # noqa: F401
from nsvla.solver.sparsify import sparsify  # noqa: F401

__all__ = [
    "ActionSolver",
    "SOLVER_REGISTRY",
    "LAZY_SOLVERS",
    "load_solver_module",
    "make_solver",
    "register_solver",
    "FakeSolver",
    "RemoteSolver",
    "PrimitiveBridge",
    "render_sub_instruction",
    "sparsify",
]


def __getattr__(name):  # pragma: no cover - the reference solver needs its own deps
    if name == "VLAAdapterSolver":
        from nsvla.solver.vla_adapter import VLAAdapterSolver

        return VLAAdapterSolver
    raise AttributeError(name)
