"""Reference toy engine: deterministic, pure-Python, zero dependencies.

It models exactly the three ingredients whose interaction decides spec-decode
correctness on hybrid GDN architectures:

  * a greedy target model  -- next token = F(recurrent state, position)
  * a recurrent state      -- folded from EVERY emitted token (GDN-like:
                              rejection must roll the state forward through
                              the target's bonus token, never drop it)
  * draft/verify           -- K drafts per step; acceptance by comparison
                              (standard, lossless) or by configured rates
                              (synthetic, vLLM 0.32 semantics)

The correct implementation keeps the model's view of the token stream in
lockstep with the emitted stream. Bug classes are introduced by subclassing
and overriding the hooks (see mutants.py); each hook documents the real-world
issue it emulates.

The batch is stepped round-robin (every active request advances one step per
round, requests finish at different rounds and leave the batch), so batch
composition, ordering and compaction dynamics are actually exercised.
"""
from __future__ import annotations

import random
from typing import Sequence

from ..scenario import (
    EngineOptions,
    ReqSpec,
    unconditional_to_conditional,
    validate_rates,
)
from .base import EngineInfo, GenResult

VOCAB = 101
_WINDOW = 64          # state hashing window (bounded for speed)
_FNV_OFFSET, _FNV_PRIME = 0x811C9DC5, 0x01000193
_CONFIDENT_MARGIN = 3.0   # far above any sane tau
_NEAR_TIE_MARGIN = 0.25   # below the default tau=0.5
_NEAR_TIE_FROM = 32       # emitted positions before this are always confident


def fnv1a(tokens: Sequence[int]) -> int:
    h = _FNV_OFFSET
    for t in tokens:
        h = ((h ^ (t & 0xFFFFFFFF)) * _FNV_PRIME) & 0xFFFFFFFF
    return h


def prompt_tokens(prompt: str) -> list[int]:
    return list(prompt.encode("utf-8"))[:512] or [0]


class _ReqRun:
    """Per-request execution state inside one generate() call."""

    def __init__(self, req: ReqSpec, seed: int):
        self.req = req
        self.prompt = prompt_tokens(req.prompt)
        self.emitted: list[int] = []        # ground-truth output stream
        self.model_tokens: list[int] = list(self.prompt)  # what the forward sees
        self.accept_lens: list[int] = []
        self.last_event: tuple[str, list[int]] = ("plain", [])
        self.rng_seed = fnv1a(
            [fnv1a(req.rid.encode("utf-8")), seed]
        ) & 0x7FFFFFFF
        self.local_step = 0

    @property
    def done(self) -> bool:
        return len(self.emitted) >= self.req.max_tokens

    def state_digest(self) -> int:
        return fnv1a(self.model_tokens)


class ReferenceEngine:
    name = "reference"

    # ---- model -------------------------------------------------------

    def _window(self, run: _ReqRun, batch_len: int) -> list[int]:
        """Token window fed to the state hash. Packed-batch padding bugs
        (vLLM #53051/#53929 class) corrupt exactly this view."""
        return run.model_tokens[-_WINDOW:]

    def _position(self, run: _ReqRun, batch_len: int) -> int:
        """Logical position of the next token. Padded-layout bugs offset it."""
        return len(run.model_tokens)

    def _target_token(self, run: _ReqRun, batch_len: int, prefix_extra: Sequence[int] = ()) -> int:
        """Greedy next token of the toy target model, given the current
        model_tokens plus any tentative draft prefix."""
        window = run.model_tokens[-_WINDOW:]
        if prefix_extra:
            window = (window + list(prefix_extra))[-_WINDOW:]
        pos = self._position(run, batch_len) + len(prefix_extra)
        d = fnv1a(window)
        return (d * 2654435761 + pos * 40503 + 7) % VOCAB

    def _draft_token(self, run: _ReqRun, batch_len: int, prefix_extra: Sequence[int] = (), mode: str = "learned") -> int:
        tgt = self._target_token(run, batch_len, prefix_extra)
        if mode == "oracle":
            return tgt
        if mode == "adversarial":
            return (tgt + 1 + (fnv1a(list(prefix_extra) or [0]) % 5)) % VOCAB
        # learned: deterministic drafter correct with prob ~2/3
        r = fnv1a([fnv1a(list(prefix_extra) or [0]), self._position(run, batch_len), 0xD]) % 3
        return tgt if r < 2 else (tgt + 1 + r) % VOCAB

    def _margin_for(self, run: _ReqRun, emitted_idx: int) -> float:
        """Deterministic per-(request, emitted position) top1-top2 gap.

        Early positions are always confident; from _NEAR_TIE_FROM on, ~20%
        of positions are near-ties. Margin-aware oracles must classify
        divergences at near-tie positions as BOUNDARY and at confident
        positions as FAIL."""
        if emitted_idx < _NEAR_TIE_FROM:
            return _CONFIDENT_MARGIN
        r = fnv1a([run.rng_seed, emitted_idx, 0x5A17]) % 5
        return _NEAR_TIE_MARGIN if r == 0 else _CONFIDENT_MARGIN

    # ---- hooks (mutants override these) --------------------------------

    def _commit(self, run: _ReqRun, new_tokens: Sequence[int]) -> None:
        """Commit emitted tokens to both streams. Rejection-path drift
        (SGLang #35150 class) drops tokens here."""
        run.emitted.extend(new_tokens)
        run.model_tokens.extend(new_tokens)
        self._truncate(run)

    @staticmethod
    def _truncate(run: _ReqRun) -> None:
        """A decode step emits up to K+1 tokens; engines stop exactly at
        max_tokens, so both streams are cut to length."""
        over = len(run.emitted) - run.req.max_tokens
        if over > 0:
            del run.emitted[-over:]
            del run.model_tokens[-over:]

    def _pre_round(self, batch: list[_ReqRun]) -> None:
        """Called before each batch round. Cross-request state pollution
        (isolation/permutation bugs) hooks here."""
        pass

    def _post_step(self, run: _ReqRun, batch_len: int, opts: EngineOptions) -> None:
        """Called after one decode step. Regime-sensitive corruption
        (CUDA-graph replay bugs) hooks here."""
        pass

    # ---- decoding ------------------------------------------------------

    def generate(self, requests: Sequence[ReqSpec], opts: EngineOptions) -> list[GenResult]:
        runs = [_ReqRun(r, opts.seed) for r in requests]
        if opts.method == "none":
            self._run_target_only(runs, opts)
        else:
            self._run_spec(runs, opts)
        return [self._result(r) for r in runs]

    def _run_target_only(self, runs: list[_ReqRun], opts: EngineOptions) -> None:
        active = [r for r in runs]
        while active:
            self._pre_round(active)
            active = [r for r in active if not r.done]
            for r in active:
                tok = self._target_token(r, len(active))
                r.last_event = ("plain", [tok])
                self._commit(r, [tok])
                self._post_step(r, len(active), opts)
                r.local_step += 1
            active = [r for r in active if not r.done]

    def _run_spec(self, runs: list[_ReqRun], opts: EngineOptions) -> None:
        k = opts.num_speculative_tokens
        if opts.rejection_sample_method == "synthetic":
            if opts.synthetic_acceptance_rates is None:
                raise ValueError("synthetic method requires synthetic_acceptance_rates")
            validate_rates(opts.synthetic_acceptance_rates, k)
            cond = unconditional_to_conditional(opts.synthetic_acceptance_rates)
        else:
            cond = None
        active = [r for r in runs]
        while active:
            self._pre_round(active)
            active = [r for r in active if not r.done]
            for r in active:
                self._spec_step(r, len(active), opts, k, cond)
                self._post_step(r, len(active), opts)
                r.local_step += 1
            active = [r for r in active if not r.done]

    def _spec_step(self, run: _ReqRun, batch_len: int, opts: EngineOptions,
                   k: int, cond: tuple[float, ...] | None) -> None:
        emitted_this_step: list[int] = []
        accepted = 0
        for j in range(k):
            draft = self._draft_token(run, batch_len, emitted_this_step, opts.draft_mode)
            if cond is not None:
                # synthetic: acceptance decided by configured conditional rate
                u = _step_rng(run, j)()
                if u >= cond[j]:
                    break
            else:
                # standard lossless verify: accept iff draft == target's choice
                tgt = self._target_token(run, batch_len, emitted_this_step)
                if draft != tgt:
                    break
            emitted_this_step.append(draft)
            accepted += 1
        # bonus token: the target's choice at the first non-accepted position
        bonus = self._target_token(run, batch_len, emitted_this_step)
        emitted_this_step.append(bonus)
        run.last_event = (
            "full_accept" if accepted == k else "reject",
            emitted_this_step,
        )
        self._commit(run, emitted_this_step)
        run.accept_lens.append(accepted + 1)

    def _result(self, run: _ReqRun) -> GenResult:
        return GenResult(
            rid=run.req.rid,
            token_ids=list(run.emitted),
            num_steps=run.local_step,
            accept_lens=list(run.accept_lens),
            final_state=run.state_digest(),
            margins=[self._margin_for(run, i) for i in range(len(run.emitted))],
        )

    def info(self) -> EngineInfo:
        return EngineInfo(
            name=self.name,
            details={
                "vocab": VOCAB,
                "impl": "pure-python",
                "capabilities": ("draft_mode", "state_digest", "graph_regime"),
            },
        )


def _step_rng(run: _ReqRun, j: int):
    """Deterministic per-(request, step, position) sampler -- independent of
    batch order and batch composition, as a correct engine must be."""
    r = random.Random(fnv1a([run.rng_seed, run.local_step, j]) & 0x7FFFFFFF)
    return r.random
