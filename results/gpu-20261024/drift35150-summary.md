# SpecSafe run `20261024_171536-drift35150`

- matrix: **drift35150**
- engine: **vllm-offline**
- started: 2026-10-24T17:15:36  duration: 601.4s
- verdict: **FAILURES PRESENT** (0 pass / 3 fail / 0 error)

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
| drift-k1-n4-L256 | all_reject_equivalence | fail | [r000] spec_forced_reject vs target_only: diverge at pos 6 (len 256 vs 256); [r001] spec_forced_reject vs target_only: diverge at pos 0 (len 256 vs 256); [r002] spec_forced_reject vs target_only: diverge at pos 2 (len 256 vs 256) … |
| drift-k3-n8-L512 | all_reject_equivalence | fail | [r002] spec_forced_reject vs target_only: diverge at pos 15 (len 512 vs 512); [r007] spec_forced_reject vs target_only: diverge at pos 0 (len 512 vs 512) |
| drift-k7-n32-L256 | all_reject_equivalence | fail | [r004] spec_forced_reject vs target_only: diverge at pos 9 (len 256 vs 256); [r006] spec_forced_reject vs target_only: diverge at pos 154 (len 256 vs 256); [r011] spec_forced_reject vs target_only: diverge at pos 0 (len 256 vs 256) … |

