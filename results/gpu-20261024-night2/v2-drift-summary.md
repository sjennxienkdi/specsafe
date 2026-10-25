# SpecSafe run `20261025_021556-drift35150`

- matrix: **drift35150**
- engine: **vllm-offline**, margin tau=0.5, 3x repeats
- started: 2026-10-25T02:15:56  duration: 1525.2s
- verdict: **FAILURES PRESENT** (0 pass / 2 boundary / 1 fail / 0 error)

Verdict legend: PASS = token-identical; BOUNDARY = divergence only at near-tie positions (margin <= tau, numeric floor); FAIL = confident divergence (margin > tau both sides).

## Engine

- vllm_version: `0.32.0`
- model: `/root/autodl-tmp/models/qwen3.8-27b-fp8`
- max_model_len: `4096`
- tp: `1`
- capabilities: `('graph_regime', 'logprob_margin')`
- resolved_spec_configs: `{"(('gpu_memory_utilization', 0.85), ('max_model_len', 4096))": {'note': 'spec config echo unavailable (internals moved)'}}`

## Results

| case | invariant | status | detail |
|---|---|---|---|
| drift-k1-n4-L256 | all_reject_equivalence | **boundary** | [repeats boundary,boundary,boundary] [r000] spec_forced_reject vs target_only: diverge at pos 1 (len 256 vs 256) margin a=0.0 b=0.125 -> BOUNDARY; [r002] spec_forced_reject vs target_only: diverge at pos 0 (len 256 vs 256) margin a=0.0625 b=0.0 -> BOUNDARY; [r003] spec_forced_reject vs target_only: diverge at pos 57 (len 256 vs 256) margin a=4.875 b=0.5 -> BOUNDARY |
| drift-k3-n8-L512 | all_reject_equivalence | **boundary** | [repeats boundary,boundary,boundary] [r002] spec_forced_reject vs target_only: diverge at pos 6 (len 512 vs 512) margin a=0.0 b=0.25 -> BOUNDARY; [r006] spec_forced_reject vs target_only: diverge at pos 241 (len 512 vs 512) margin a=0.5 b=5.3125 -> BOUNDARY; [r007] spec_forced_reject vs target_only: diverge at pos 0 (len 512 vs 512) margin a=0.125 b=0.0 -> BOUNDARY |
| drift-k7-n32-L256 | all_reject_equivalence | **fail** | [repeats fail,fail,fail] [r004] spec_forced_reject vs target_only: diverge at pos 9 (len 256 vs 256) margin a=0.0625 b=0.0 -> BOUNDARY; [r006] spec_forced_reject vs target_only: diverge at pos 154 (len 256 vs 256) margin a=2.25 b=1.375 -> FAIL; [r014] spec_forced_reject vs target_only: diverge at pos 10 (len 256 vs 256) margin a=0.0 b=0.125 -> BOUNDARY … |

