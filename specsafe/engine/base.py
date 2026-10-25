"""Engine adapter interface shared by the reference toy engine and vLLM."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, Sequence

from ..scenario import EngineOptions, ReqSpec


@dataclass
class GenResult:
    rid: str
    token_ids: list[int]            # generated tokens, prompt excluded
    num_steps: int = 0
    accept_lens: list[int] | None = None   # per decode step: emitted tokens (drafts accepted + bonus)
    final_state: int | None = None         # recurrent-state digest, if the engine exposes it
    margins: list[float] | None = None     # top1-top2 logprob margin per token, if available


@dataclass
class EngineInfo:
    name: str
    details: dict = field(default_factory=dict)   # versions, config echo, ...


class Engine(Protocol):
    """A stateless-across-calls decoding engine.

    generate() runs one batch of requests under opts and returns one GenResult
    per request, in the same order as the input. Implementations must be
    deterministic for fixed (requests, opts).
    """

    def generate(self, requests: Sequence[ReqSpec], opts: EngineOptions) -> list[GenResult]: ...

    def info(self) -> EngineInfo: ...
