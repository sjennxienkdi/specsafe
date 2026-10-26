# SpecSafe

**English** | [简体中文](README.zh-CN.md)

**Differential correctness testing for speculative decoding** — targeting hybrid
GDN/Attention architectures (Qwen3.5 / Qwen3.8 / Qwen3-Next family) on vLLM/SGLang.
It auto-generates accept/reject/mixed scenarios, checks six behavioral invariants,
and detects state drift, graph-geometry mismatches, and batch-reordering errors —
with a margin-aware oracle that separates real bugs from the numeric
nondeterminism floor.

## Why it exists

Speculative-decoding speedup numbers are everywhere; correctness auditing is
close to blank:

- vLLM's own [RFC #28135](https://github.com/vllm-project/vllm/issues/28135)
  (2025-11, spec-decode CI regression testing) admits existing e2e tests use
  arbitrary 66–80% text-match thresholds with no acceptance tracking — the RFC
  went stale-closed, never implemented
- [SGLang #35150](https://github.com/sgl-project/sglang/issues/35150) (open,
  2026-08): Qwen3.8 + DSpark **forced-reject is not lossless** — outputs drift
  with accumulated GDN state when drafts are force-rejected; root cause never
  isolated
- [vLLM #53051](https://github.com/vllm-project/vllm/issues/53051) (open):
  prefill entering a spec-decode CUDA graph → **silent** GDN state loss and
  garbled output
- [vLLM #53929](https://github.com/vllm-project/vllm/pull/53929) (open PR):
  captured uniform layout overwritten by ragged replay

The common thread: **these are not unit-testable feature bugs — they are broken
state-machine/layout invariants.** SpecSafe turns those invariants into
reusable differential tests.

## First real catch: vLLM #63593 (2026-10-25)

SpecSafe's margin-aware verdict separated **one real state-corruption bug** from
36 numeric-noise alarms on vLLM 0.32.0 + Qwen3.8-27B-FP8, then minimized it:

- **Symptom**: MTP speculation with forced-reject diverges from target-only with
  two-sided confident margins of 0.6–5.9 nats (this stack's measured noise floor
  is ≤0.5 nats) and a fully decorrelated tail — GDN state contaminated by
  rejected drafts
- **Minimal reproducer**: K≥5, n≥8, L≥192 (single-axis greedy reduction,
  S0–S8 grid, 3 repeats × 2 independent boots, stable); the victim request is
  deterministic (same prompt every time — bisectable) and the first-divergence
  position moves with batch size (n=8 → pos 154, n=32 → pos 181)
- **Classification**: a new MTP member of the
  [#39273](https://github.com/vllm-project/vllm/issues/39273) GDN
  reject-state-corruption family — that thread had explicitly considered the MTP
  path clean
- **Report**: [vllm#63593](https://github.com/vllm-project/vllm/issues/63593)
  (one-command ~3-min regression case, boot-stable — unlike the family head
  #39273, which does not reproduce across restarts)

Without margin awareness this bug is indistinguishable from noise — which is
the reason this tool exists.

## The six invariants

| # | Invariant | Bug class it catches |
|---|---|---|
| 1 | **all-reject equivalence**: all drafts rejected ⇒ output == target-only decoding | reject-path state rollback (#35150) |
| 2 | **accept-path equivalence**: any accept pattern under lossless verification ⇒ output == target-only | accept-path state folding / prefix misalignment |
| 3 | **permutation invariance**: batch reordering changes no request's output | slot indexing, compaction reordering |
| 4 | **split/merge equivalence**: packed execution == per-request independent execution | ragged/padded layouts (#53051) |
| 5 | **state isolation**: A's accepts/rejects never affect B's output or state | recurrent/conv state cross-contamination |
| 6 | **graph geometry consistency**: graph regime switching (captured/eager) changes no output | capture/replay layout mismatch (#53929) |

## Quick start (local, zero deps, no GPU)

```bash
python -m specsafe.cli selftest     # double-line-of-defense proof (below)
python -m specsafe.cli run --engine reference --matrix smoke --repeats 3 --out results/smoke-ref
python -m specsafe.cli gen --matrix static32 --out scenarios/static32.json
```

`selftest` is the oracle's **double line of defense**:

1. Six mutant engines (each simulating a real bug class: #35150 reject-path
   drift, #53051 silent graph corruption, …) all **FAIL** under the
   margin-aware verdict — the oracle does not launder real bugs
2. A NoisyFlip test double (flips tokens only at near-tie positions, simulating
   hardware tie-break variance) is judged **BOUNDARY** at τ=0.5 and FAIL at
   τ=0 — the threshold actually discriminates

## Verdict semantics (oracle v2, three states)

| Verdict | Meaning |
|---|---|
| **PASS** | token streams identical across arms |
| **BOUNDARY** | divergence exists, but every divergence position has a top1–top2 logprob margin ≤ τ on at least one side (numeric boundary, explained by the noise floor — not guilty) |
| **FAIL** | confident divergence with two-sided margin > τ — a semantic difference noise cannot explain |

Divergence without margin evidence is conservatively judged FAIL. Each case can
run `--repeats N` with majority verdicts (fail > boundary > pass, worst wins
ties). τ defaults to 0.5 nats, calibrated by the rebuild-pair `floor` matrix.

## Cloud (real vLLM runs)

See `cloud/RUNBOOK.md` (upload / preflight / matrices / #35150 shapes /
teardown, fully scripted):

```bash
python3 -m specsafe.cli run --engine vllm --matrix static32 \
    --model ~/models/qwen3.8-27b-fp8 --spec-method mtp \
    --max-model-len 4096 --out results/static32-vllm-fp8
```

## Semantics (important)

- **Standard (lossless) verification**: under greedy, spec output **must**
  equal target-only token for token regardless of accept/reject pattern — the
  basis for invariants 1/2
- **Synthetic rates** (vLLM 0.32 `rejection_sample_method="synthetic"`):
  `rates[i]` = marginal probability that the first i+1 drafts are all accepted.
  `[0.0]*K` forces all-reject every step — each emitted token is still the
  target's greedy choice, so output equivalence remains a hard requirement (the
  #35150 property); `[1.0]*K` accepts draft tokens (≠ target choices) and is
  not used for equivalence checks
- The reference engine supports `draft_mode=oracle/adversarial` to construct
  exact all-accept/all-reject; vLLM's public API cannot force draft quality, so
  the checker auto-degrades to the synthetic arm

## Architecture

```
specsafe/
  scenario.py    # versioned scenarios (JSON-serializable; local gen == cloud replay)
  generator.py   # deterministic matrices: selftest / smoke / static32 / drift35150 / floor
  invariants.py  # six invariant checkers (behavioral layer, engine-agnostic)
  engine/
    reference.py    # pure-Python deterministic toy GDN engine (state fold + reject rollback)
    mutants.py      # six mutant engines, one per real bug class
    vllm_offline.py # vLLM 0.32 offline-API adapter (field names verified against source)
  report.py      # report.json + summary.md matrix reports
  cli.py         # selftest / run / gen
```

Tiered design: the behavioral layer (output comparison) runs first; state-level
probes (GDN state norm/max-diff, requires vLLM instrumentation) are enabled on
demand — only escalated when the behavioral layer shows drift.

## Version compatibility

| Engine | Version | Status |
|---|---|---|
| vLLM | 0.32.0 clean + torch 2.13.0 (2026-10-25) | all six invariants hold above the noise floor (static32: 8 pass / 26 boundary / 0 fail ×3); **one stable confident divergence minimized to K≥5/n≥8/L≥192 → [#63593](https://github.com/vllm-project/vllm/issues/63593)** |
| vLLM | 0.32.0 stock + torch 2.12 (2026-10-24) | numeric nondeterminism floor discovered and calibrated; token-exact equality unusable |
| SGLang | corresponding | planned (#35150 as-is reproduction needs SGLang+NVFP4+DSpark) |

**Boundary map v1** (RTX PRO 6000 Blackwell 96GB × Qwen3.8-27B-FP8): rebuild-pair
floor margin ≤0.1875 (50% exact ties); batch-composition floor ≤0.125; all
static-path divergences ≤ τ=0.5. τ keeps a 2.7× safety margin over the measured
floor.

## Field finding: the numeric nondeterminism floor of vLLM 0.32.0 (2026-10-24)

On RTX PRO 6000 × Qwen3.8-27B-FP8 × vLLM 0.32.0 (stock), token-exact equality
flagged 37 of 43 cases as FAIL. **The control probe found the root cause**
(`results/gpu-20261024/probe_pos0.json`):

- target-only rebuilt twice with identical config diverges **by itself**
  (r003@pos17, margin = 0.0 — a complete logit tie broken arbitrarily)
- spec-vs-target divergences (margins 0.0–0.375) all fall inside that noise floor
- conclusion: a near-tie argmax floor exists that flips with engine build and
  batch composition, independent of speculation — token-exact equality is not a
  usable oracle on such stacks. This is where the three-state verdict and the
  `floor` matrix (rebuild-pair calibration + batch-composition localization)
  come from

We are not aware of a public precedent for this data (the "66% match threshold"
criticized in RFC #28135 is what the absence of a floor concept looks like).

## Known boundaries

- The offline API cannot exercise runtime insertion/cancellation/compaction
  lifecycle events — a planned server-mode adapter (continuous submission +
  dynamic arrival) covers this
- The state layer (final_state) is exposed by the reference engine only; real
  engines need instrumentation (W4 tiered design: escalate only when the
  behavioral layer shows real divergence)
- τ portability: the 0.5-nat starting value comes from one GPU/model's rebuild
  pairs; other hardware/precision needs recalibration via the `floor` matrix
  (that's a feature — the floor itself is a measurable object)
