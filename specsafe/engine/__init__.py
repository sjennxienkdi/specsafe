"""Engine registry."""
from __future__ import annotations

from .base import Engine, EngineInfo, GenResult
from .reference import ReferenceEngine


def make_engine(name: str, **kwargs):
    """name: 'reference' | 'mutant:<class>' | 'vllm'."""
    if name == "reference":
        return ReferenceEngine()
    if name.startswith("mutant:"):
        from . import mutants
        cls = getattr(mutants, name.split(":", 1)[1], None)
        if cls is None:
            raise KeyError(f"unknown mutant '{name}'")
        return cls()
    if name == "vllm":
        from .vllm_offline import VllmEngine
        return VllmEngine(**kwargs)
    raise KeyError(f"unknown engine '{name}'")


__all__ = ["Engine", "EngineInfo", "GenResult", "ReferenceEngine", "make_engine"]
