"""vLLM 0.32.x offline-API adapter.  *** CLOUD-BOX ONLY — unverified locally ***

Written against v0.32.0 source (vllm-src/): speculative_config accepts
method / num_speculative_tokens / rejection_sample_method /
synthetic_acceptance_rates (per-position unconditional rates, non-increasing,
len == K) / draft_sample_method / num_speculative_tokens_per_batch_size.

Mapping from EngineOptions:
  method                     -> speculative_config["method"]        ("mtp", "dspark", ...)
  num_speculative_tokens     -> speculative_config["num_speculative_tokens"]
  rejection_sample_method    -> speculative_config["rejection_sample_method"]
  synthetic_acceptance_rates -> speculative_config["synthetic_acceptance_rates"]
  graph_regime "eager"       -> enforce_eager=True; "captured" -> default graphs
  draft_mode                 -> NOT SUPPORTED by public API (warns; checkers
                                fall back to synthetic-rate arms)
  extra kv starting "--"     -> passed through to LLM(**kw) via _extra_llm_kwargs

A separate vLLM engine is constructed per distinct speculative config and
kept cached; call shutdown() to release VRAM before constructing differently
configured engines (the runbook orders runs so at most one big engine is
resident at a time).
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Sequence

from ..scenario import EngineOptions, ReqSpec
from .base import EngineInfo, GenResult


class VllmEngine:
    name = "vllm"

    def __init__(self, model: str, max_model_len: int = 2048,
                 tensor_parallel_size: int = 1,
                 gpu_memory_utilization: float = 0.85,
                 max_num_seqs: int = 64,
                 extra_llm_kwargs: dict | None = None,
                 default_spec_method: str = "mtp",
                 quiet: bool = True):
        try:
            import vllm  # noqa: F401
            from vllm import LLM, SamplingParams  # noqa: F401
        except ImportError as e:  # pragma: no cover - cloud box only
            raise RuntimeError(
                "vllm is not importable; the vLLM adapter runs on the cloud "
                "box (or any machine with a vLLM install), not on the local "
                "dev machine."
            ) from e
        if quiet:
            os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
        self._vllm = vllm
        self.model = model
        self.max_model_len = max_model_len
        self.tp = tensor_parallel_size
        self.gpu_mem = gpu_memory_utilization
        # GDN/Mamba hybrids need one state-cache block per decode sequence;
        # vLLM's default max_num_seqs=1024 can exceed available blocks
        # (v0.32.0: "max_num_seqs exceeds available Mamba cache blocks").
        # SpecSafe matrices never exceed ~64 concurrent requests.
        self.max_num_seqs = max_num_seqs
        self.default_spec_method = default_spec_method
        # Single-resident policy: each vLLM engine reserves
        # gpu_memory_utilization of the card; two 0.85-util engines cannot
        # coexist on one GPU. See _release_all for the eviction semantics.
        self.extra_llm_kwargs = dict(extra_llm_kwargs or {})
        self._engines: dict[tuple, object] = {}   # key -> engine, LRU order
        self._warned: set[str] = set()
        self._resolved_configs: dict[tuple, dict] = {}

    # ------------------------------------------------------------ helpers

    def _llm_kwargs(self, opts: EngineOptions) -> dict:
        kw = dict(
            model=self.model,
            max_model_len=self.max_model_len,
            tensor_parallel_size=self.tp,
            gpu_memory_utilization=self.gpu_mem,
            max_num_seqs=self.max_num_seqs,
        )
        if opts.method != "none":
            # "spec" is the engine-agnostic alias; real engines need a method
            method = (opts.method if opts.method != "spec"
                      else self.default_spec_method)
            spec = {
                "method": method,
                "num_speculative_tokens": opts.num_speculative_tokens,
            }
            if opts.rejection_sample_method == "synthetic":
                spec["rejection_sample_method"] = "synthetic"
                spec["synthetic_acceptance_rates"] = [
                    float(r) for r in (opts.synthetic_acceptance_rates or ())
                ]
            # opts.extra keys with a spec_ prefix flow into speculative_config
            # (e.g. spec_model=<drafter> for dspark)
            for k, v in opts.extra:
                if k.startswith("spec_"):
                    spec[k[len("spec_"):]] = _parse_extra(v)
            kw["speculative_config"] = spec
        if opts.graph_regime == "eager":
            kw["enforce_eager"] = True
        for k, v in opts.extra:
            # spec_ prefix -> speculative_config keys (e.g. spec_model);
            # probe_ prefix -> metadata only: it distinguishes engine-cache
            # keys (forcing independent rebuilds for floor measurement)
            # without reaching the LLM constructor.
            if not k.startswith("spec_") and not k.startswith("probe_"):
                kw[k] = _parse_extra(v)
        kw.update(self.extra_llm_kwargs)
        return kw

    def _engine_for(self, opts: EngineOptions):
        from vllm import LLM
        kw = self._llm_kwargs(opts)
        key = _freeze(kw)
        if key in self._engines:
            return self._engines[key]
        self._release_all()          # single-resident policy
        llm = LLM(**kw)
        self._engines[key] = llm
        self._resolved_configs[key] = _resolved_spec_echo(llm)
        return llm

    def _release_all(self) -> None:
        """Drop cached engines and make sure their EngineCore subprocesses
        actually release VRAM (del+gc is not enough in v0.32: EngineCore is
        a separate process)."""
        import gc
        import subprocess
        import time

        if not self._engines:
            return
        self._engines.clear()
        gc.collect()
        for _ in range(12):                      # up to 60s graceful
            time.sleep(5)
            used = _gpu_used_mib()
            if used is not None and used < 4000:
                return
        # all EngineCores on this box belong to us (neighbor queue gated out)
        subprocess.run(["pkill", "-9", "-f", "EngineCor[e]"], check=False)
        for _ in range(6):
            time.sleep(5)
            used = _gpu_used_mib()
            if used is not None and used < 4000:
                return

    def shutdown(self) -> None:
        self._release_all()

    # ------------------------------------------------------------ Engine API

    def generate(self, requests: Sequence[ReqSpec], opts: EngineOptions) -> list[GenResult]:
        from vllm import SamplingParams
        if opts.draft_mode != "learned" and "draft_mode" not in self._warned:
            self._warned.add("draft_mode")
            warnings.warn(
                "vLLM public API cannot force draft quality (draft_mode); "
                "checkers fall back to synthetic-rate arms", stacklevel=2)
        llm = self._engine_for(opts)
        sps = [
            SamplingParams(temperature=0.0, max_tokens=r.max_tokens,
                           seed=opts.seed, logprobs=2)
            for r in requests
        ]
        outs = llm.generate([r.prompt for r in requests], sps)
        results = []
        for req, out in zip(requests, outs):
            co = out.outputs[0]
            margins = _margins(co)
            results.append(GenResult(
                rid=req.rid,
                token_ids=list(co.token_ids),
                num_steps=len(co.token_ids),          # per-token steps unknown offline
                accept_lens=_accept_lens(out),
                final_state=None,                     # state layer needs instrumentation
                margins=margins,
            ))
        return results

    def info(self) -> EngineInfo:
        v = self._vllm.__version__
        return EngineInfo(
            name=f"{self.name}-offline",
            details={
                "vllm_version": v,
                "model": self.model,
                "max_model_len": self.max_model_len,
                "tp": self.tp,
                "capabilities": ("graph_regime", "logprob_margin"),
                "resolved_spec_configs": {
                    str(k[:2]): c for k, c in self._resolved_configs.items()
                },
            },
        )


# ------------------------------------------------------------------ utils

def _gpu_used_mib() -> int | None:
    import subprocess
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001 - environment probe, best effort
        return None


def _parse_extra(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        return v


def _freeze(kw: dict) -> tuple:
    return tuple(sorted((k, _hashable(v)) for k, v in kw.items()))


def _hashable(v):
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    return v


def _margins(completion_output) -> list[float] | None:
    lps = getattr(completion_output, "logprobs", None)
    if not lps:
        return None
    out = []
    for pos in lps:
        if not pos:
            out.append(float("nan"))
            continue
        vals = sorted((lp.logprob for lp in pos.values()), reverse=True)
        out.append(round(vals[0] - vals[1], 4) if len(vals) > 1 else float("inf"))
    return out


def _accept_lens(request_output) -> list[int] | None:
    m = getattr(request_output, "metrics", None)
    if m is None:
        return None
    for attr in ("acceptance_lengths", "spec_acceptance_lengths",
                 "accepted_lengths"):
        v = getattr(m, attr, None)
        if v:
            return list(v)
    return None


def _resolved_spec_echo(llm) -> dict:
    """Best-effort echo of the engine's resolved speculative config, for the
    report's environment fingerprint (defensive across vLLM internals)."""
    for holder in ("llm_engine", "vllm_engine", "engine"):
        eng = getattr(llm, holder, None)
        if eng is None:
            continue
        cfg = getattr(eng, "speculative_config", None)
        if cfg is not None:
            return {
                a: getattr(cfg, a)
                for a in ("method", "num_speculative_tokens",
                          "rejection_sample_method",
                          "synthetic_acceptance_rates")
                if getattr(cfg, a, None) is not None
            }
    return {"note": "spec config echo unavailable (internals moved)"}
