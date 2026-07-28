"""Repo-relative filesystem defaults, each overridable by an environment variable.

Nothing in the codebase hard-codes an absolute path. Every root below defaults to a
directory inside the repository and is redirected with a single environment variable,
which is how large artifacts (datasets, checkpoints, cached features, run outputs) are
moved onto a data volume without touching any config file.

    NSVLA_DATA_ROOT        datasets and derived data          default: data
    NSVLA_FEATURE_ROOT     cached frozen-VLM features         default: data/features
    NSVLA_ANNOTATION_ROOT  primitive segment annotations      default: data/primitive_annotations
    NSVLA_MODEL_ROOT       model weights / checkpoints        default: checkpoints
    NSVLA_RUN_ROOT         experiment outputs                 default: runs
    NSVLA_LIBERO_ROOT      LIBERO source tree                 default: third_party/LIBERO
    NSVLA_LIBERO_PLUS_ROOT LIBERO-Plus source tree            default: third_party/LIBERO-Plus
    NSVLA_SOLVER_ROOT      action-solver source tree          default: third_party/VLA-Adapter
    NSVLA_VLM_MODEL        frozen VLM encoder weights         default: <NSVLA_MODEL_ROOT>/qwen3-vl-2b
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def root(var: str, default: str) -> str:
    """Absolute path for ``var``, falling back to ``default`` under the repo root."""
    value = os.environ.get(var)
    if value:
        return str(Path(value).expanduser())
    return str(REPO_ROOT / default)


def data_root() -> str:
    return root("NSVLA_DATA_ROOT", "data")


def feature_root() -> str:
    return root("NSVLA_FEATURE_ROOT", "data/features")


def annotation_root() -> str:
    return root("NSVLA_ANNOTATION_ROOT", "data/primitive_annotations")


def model_root() -> str:
    return root("NSVLA_MODEL_ROOT", "checkpoints")


def run_root() -> str:
    return root("NSVLA_RUN_ROOT", "runs")


def libero_root() -> str:
    return root("NSVLA_LIBERO_ROOT", "third_party/LIBERO")


def libero_plus_root() -> str:
    return root("NSVLA_LIBERO_PLUS_ROOT", "third_party/LIBERO-Plus")


def solver_root() -> str:
    """Source tree of the reference action solver (VLA-Adapter)."""
    return root("NSVLA_SOLVER_ROOT", "third_party/VLA-Adapter")


def vlm_model_dir() -> str:
    """Frozen VLM encoder weights."""
    return os.environ.get("NSVLA_VLM_MODEL", str(Path(model_root()) / "qwen3-vl-2b"))
