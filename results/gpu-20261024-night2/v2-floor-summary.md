# SpecSafe run `20261024_221253-floor`

- matrix: **floor**
- engine: **vllm-offline**, margin tau=0.5, 1x repeats
- started: 2026-10-24T22:12:53  duration: 109.9s
- verdict: **ALL PASS** (0 pass / 2 boundary / 0 fail / 0 error)

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
| floor-rebuild-n3 | rebuild_floor | **boundary** | [r000] build0 vs build1: diverge at pos 1 (len 64 vs 64) margin a=0.0 b=0.125 -> BOUNDARY; [r000] build0 vs build2: diverge at pos 1 (len 64 vs 64) margin a=0.0 b=0.1875 -> BOUNDARY |
| floor-batch-n8 | batch_floor | **boundary** | [r002] packed vs per_request: diverge at pos 0 (len 96 vs 96) margin a=0.0 b=0.0625 -> BOUNDARY; [r003] packed vs per_request: diverge at pos 1 (len 96 vs 96) margin a=0.0625 b=0.0 -> BOUNDARY; [r004] packed vs per_request: diverge at pos 6 (len 96 vs 96) margin a=0.0625 b=0.0625 -> BOUNDARY … |

