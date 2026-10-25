# SpecSafe run `20261024_154343-smoke`

- matrix: **smoke**
- engine: **vllm-offline**
- started: 2026-10-24T15:43:43  duration: 589.6s
- verdict: **FAILURES PRESENT** (0 pass / 6 fail / 0 error)

## Engine

- vllm_version: `0.32.0`
- model: `/root/autodl-tmp/models/qwen3.8-27b-fp8`
- max_model_len: `4096`
- tp: `1`
- capabilities: `('graph_regime', 'logprob_margin')`
- resolved_spec_configs: `{"(('gpu_memory_utilization', 0.85), ('max_model_len', 4096))": {'note': 'spec config echo unavailable (internals moved)'}, "(('enforce_eager', True), ('gpu_memory_utilization', 0.85))": {'note': 'spec config echo unavailable (internals moved)'}}`

## Results

| case | invariant | status | detail |
|---|---|---|---|
| smoke-k1-all_reject_equivalence | all_reject_equivalence | fail | [r000] spec_forced_reject vs target_only: diverge at pos 0 (len 48 vs 48) |
| smoke-k1-accept_path_equivalence | accept_path_equivalence | fail | [r000] spec_full_accept vs target_only: diverge at pos 0 (len 48 vs 48); [r000] spec_natural vs target_only: diverge at pos 2 (len 48 vs 48) |
| smoke-k1-permutation_invariance | permutation_invariance | fail | [r000] order0 vs order1: diverge at pos 0 (len 48 vs 48) |
| smoke-k1-split_merge_equivalence | split_merge_equivalence | fail | [r000] packed vs per_request: diverge at pos 0 (len 48 vs 48) |
| smoke-k1-state_isolation | state_isolation | fail | [r000] pair[r000+r001] vs alone:r000: diverge at pos 1 (len 48 vs 48); [r000] pair[r001+r000] vs alone:r000: diverge at pos 0 (len 48 vs 48); [r000] pair[r000+r002] vs alone:r000: diverge at pos 0 (len 48 vs 48) … |
| smoke-k1-graph_geometry_consistency | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 3 (len 48 vs 48); [r000] captured vs target_only: diverge at pos 3 (len 48 vs 48) |

