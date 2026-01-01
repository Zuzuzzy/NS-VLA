"""ZMQ server exposing any ``ActionSolver`` to ``nsvla.solver.remote.RemoteSolver``.

Solver-agnostic: it owns a socket and a dispatch table, and forwards to the
``ActionSolver`` it was handed. Launch it with ``scripts/solver_server.py``.

Wire contract
-------------
Action request::

    {"full_image": HxWx3 uint8, "wrist_image": HxWx3 uint8,
     "instruction": str, "proprio": [d] float}
    -> {"action_chunk": [H, action_dim] float}

Sampling request (Stage-II online RL) adds ``{"sample": true, "sigma": float,
"seed": int|null}`` and returns ``normalized_chunk``, ``mean_chunk`` and
``sample_id`` alongside ``action_chunk``.

Control requests carry a ``cmd`` key: ``ping``, ``awr_update``, ``clear_samples``,
``save``, ``load``. Control replies always carry ``ok``; a failed action request
replies with an ``error`` string plus a zero chunk, so a REQ peer is never left
hanging on a solver-side exception.
"""
from __future__ import annotations

import time

import numpy as np

from nsvla.solver.base import ActionSolver


def serve(solver: ActionSolver, host: str = "127.0.0.1", port: int = 5678,
          verbose: bool = True) -> None:
    """Block, serving ``solver`` on ``tcp://host:port`` until interrupted."""
    import json_numpy
    import zmq

    # Fail before binding: a server that accepts connections and only then discovers it
    # cannot load its weights is worse than one that never starts.
    solver.load()

    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://{host}:{port}")
    print(f"[solver-server] {type(solver).__name__} listening on tcp://{host}:{port}",
          flush=True)

    n_req = 0
    try:
        while True:
            req = json_numpy.loads(sock.recv().decode("utf-8"))
            cmd = req.get("cmd") if isinstance(req, dict) else None

            if cmd is not None:
                try:
                    if cmd == "ping":
                        reply = {"ok": True, **solver.ping()}
                    elif cmd == "awr_update":
                        reply = {"ok": True, **solver.awr_update(
                            req["items"], beta_l=req.get("beta_l", 1.0), lr=req.get("lr"))}
                    elif cmd == "clear_samples":
                        reply = {"ok": True, **solver.clear_samples()}
                    elif cmd == "save":
                        reply = {"ok": True, **solver.save_parameters(
                            req["path"], with_optim=req.get("with_optim", True))}
                    elif cmd == "load":
                        reply = {"ok": True, **solver.load_parameters(req["path"])}
                    else:
                        raise ValueError(f"unknown cmd {cmd!r}")
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    reply = {"ok": False, "error": repr(e)}
                sock.send(json_numpy.dumps(reply).encode("utf-8"))
                continue

            n_req += 1
            t0 = time.time()
            try:
                kwargs = dict(
                    image=req["full_image"],
                    instruction=req["instruction"],
                    proprio=req["proprio"],
                    wrist_image=req.get("wrist_image"),
                )
                if req.get("sample"):
                    out = solver.act_sample(
                        sigma=float(req.get("sigma", 0.0)), seed=req.get("seed"), **kwargs)
                    reply = {
                        "action_chunk": out["action_chunk"],
                        "normalized_chunk": out["normalized_chunk"],
                        "mean_chunk": out["mean_chunk"],
                        "sample_id": out["sample_id"],
                    }
                else:
                    reply = {"action_chunk": solver.act(**kwargs)}
                if verbose:
                    print(f"[solver-server] #{n_req} {(time.time() - t0) * 1e3:6.1f}ms "
                          f"instr={str(req.get('instruction', ''))[:60]!r}", flush=True)
            except Exception as e:
                import traceback
                traceback.print_exc()
                reply = {
                    "error": repr(e),
                    "action_chunk": np.zeros((solver.chunk_H, solver.action_dim),
                                             dtype=np.float32),
                }
            sock.send(json_numpy.dumps(reply).encode("utf-8"))
    except KeyboardInterrupt:
        print("\n[solver-server] shutting down", flush=True)
    finally:
        sock.close(linger=0)
        ctx.term()
        solver.close()
