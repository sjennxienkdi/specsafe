"""Scenario spec: a versioned, JSON-serializable description of one test arm setup.

A Scenario fixes everything the behavioral layer needs to replay a run
identically on any engine: prompts, generation lengths, spec-decode options,
seeds, batch orderings. Engines translate EngineOptions into their native
knobs (the reference engine reads them directly; the vLLM adapter maps them
onto speculative_config fields).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ReqSpec:
    rid: str
    prompt: str
    max_tokens: int

    def to_dict(self) -> dict:
        return {"rid": self.rid, "prompt": self.prompt, "max_tokens": self.max_tokens}

    @staticmethod
    def from_dict(d: dict) -> "ReqSpec":
        return ReqSpec(d["rid"], d["prompt"], d["max_tokens"])


@dataclass(frozen=True)
class EngineOptions:
    """Engine-agnostic decoding knobs.

    method: "none" (target-only) or a spec method name ("spec" for the
        reference engine; "mtp"/"dspark"/"ngram" for real engines).
    num_speculative_tokens (K): draft length per step.
    rejection_sample_method: "standard" = lossless verify (output MUST equal
        target-only under greedy); "synthetic" = accept per configured rates
        (output intentionally deviates -- used to force accept/reject shapes).
    synthetic_acceptance_rates: per-position *unconditional* acceptance
        probabilities, matching vLLM 0.32 semantics: entry i = P(first i+1
        drafts all accepted); must be non-increasing, len == K.
        rates=[0.0]*K forces every draft rejected -> output must equal
        target-only. rates=[1.0]*K accepts every draft.
    draft_mode: reference-engine only. "learned" (~2/3 accuracy),
        "oracle" (draft == target, forces full-accept under standard),
        "adversarial" (draft != target, forces all-reject under standard).
    graph_regime: "captured" (CUDA-graph style) or "eager"; reference engine
        treats both as correct, mutants may not.
    """

    method: str = "none"
    num_speculative_tokens: int = 0
    rejection_sample_method: str = "standard"
    synthetic_acceptance_rates: tuple[float, ...] | None = None
    draft_mode: str = "learned"
    graph_regime: str = "captured"
    seed: int = 0
    extra: tuple[tuple[str, str], ...] = ()  # flat str->str; adapters may parse

    def with_(self, **kw) -> "EngineOptions":
        d = dict(self.__dict__)
        d.update(kw)
        return EngineOptions(**d)

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        if self.synthetic_acceptance_rates is not None:
            d["synthetic_acceptance_rates"] = list(self.synthetic_acceptance_rates)
        d["extra"] = dict(self.extra)
        return d

    @staticmethod
    def from_dict(d: dict) -> "EngineOptions":
        d = dict(d)
        if d.get("synthetic_acceptance_rates") is not None:
            d["synthetic_acceptance_rates"] = tuple(d["synthetic_acceptance_rates"])
        d["extra"] = tuple(sorted((d.get("extra") or {}).items()))
        return EngineOptions(**d)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    description: str
    requests: tuple[ReqSpec, ...]
    spec_opts: EngineOptions           # the speculative arm under test
    permutations: tuple[tuple[int, ...], ...] = ()  # batch orderings to try (I3)
    seed: int = 0

    def to_dict(self) -> dict:
        return {
            "format_version": 1,
            "scenario_id": self.scenario_id,
            "description": self.description,
            "requests": [r.to_dict() for r in self.requests],
            "spec_opts": self.spec_opts.to_dict(),
            "permutations": [list(p) for p in self.permutations],
            "seed": self.seed,
        }

    @staticmethod
    def from_dict(d: dict) -> "Scenario":
        return Scenario(
            scenario_id=d["scenario_id"],
            description=d["description"],
            requests=tuple(ReqSpec.from_dict(r) for r in d["requests"]),
            spec_opts=EngineOptions.from_dict(d["spec_opts"]),
            permutations=tuple(tuple(p) for p in d.get("permutations", [])),
            seed=d.get("seed", 0),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "Scenario":
        return Scenario.from_dict(json.loads(s))


@dataclass(frozen=True)
class CaseSpec:
    """One matrix cell: a scenario plus the invariant that judges it."""

    case_id: str
    invariant: str          # key of specsafe.invariants.CHECKERS
    scenario: Scenario

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "invariant": self.invariant,
            "scenario": self.scenario.to_dict(),
        }

    @staticmethod
    def from_dict(d: dict) -> "CaseSpec":
        return CaseSpec(d["case_id"], d["invariant"], Scenario.from_dict(d["scenario"]))


def validate_rates(rates: tuple[float, ...], k: int) -> None:
    """Mirror of vLLM SpeculativeConfig._resolve_synthetic_acceptance_rates."""
    if len(rates) != k:
        raise ValueError(f"rates must have length {k}, got {rates}")
    if not all(0.0 <= r <= 1.0 for r in rates):
        raise ValueError(f"rates entries must be in [0, 1], got {rates}")
    if any(rates[i] > rates[i - 1] for i in range(1, len(rates))):
        raise ValueError(f"rates must be non-increasing, got {rates}")


def unconditional_to_conditional(uncond: tuple[float, ...]) -> tuple[float, ...]:
    """vLLM semantics: cond[i] = uncond[i] / uncond[i-1] (uncond[-1] := 1)."""
    out, prev = [], 1.0
    for u in uncond:
        if prev <= 0.0:
            out.append(0.0)
        else:
            out.append(min(1.0, u / prev))
        prev = u
    return tuple(out)
