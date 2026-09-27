"""Offline understanding using the frozen MoReAct CVAE. T5 loads only on demand."""
from .tokenizer import SharedMotionTokenizer

__all__ = ["SharedMotionTokenizer", "InteractionCaptioner"]


def __getattr__(name):
    if name == "InteractionCaptioner":
        from .language import InteractionCaptioner
        return InteractionCaptioner
    raise AttributeError(name)
