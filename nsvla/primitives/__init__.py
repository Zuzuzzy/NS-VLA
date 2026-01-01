"""Primitive definitions and the plan-constrained primitive classifier.

The vocabulary and plan types live here (``vocab``); the classifier that infers the
active primitive under the monotone pointer is re-exported from ``nsvla.encoder``,
where it sits next to the features it consumes.
"""
from nsvla.primitives.vocab import (  # noqa: F401
    DEFAULT_OPS,
    NOOP_OP,
    PAD_OP,
    Plan,
    Primitive,
    PrimitiveVocab,
    default_vocab,
)


def __getattr__(name):  # pragma: no cover - convenience re-export, torch is lazy
    if name == "PrimitiveClassifier":
        from nsvla.encoder.primitive_clf import PrimitiveClassifier

        return PrimitiveClassifier
    raise AttributeError(name)
