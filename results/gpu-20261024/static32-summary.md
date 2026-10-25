# SpecSafe run `20261024_155341-static32`

- matrix: **static32**
- engine: **vllm-offline**
- started: 2026-10-24T15:53:41  duration: 4904.8s
- verdict: **FAILURES PRESENT** (6 pass / 28 fail / 0 error)

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
| i1-forced-reject-k1-s | all_reject_equivalence | pass |  |
| i1-forced-reject-k1-l | all_reject_equivalence | fail | [r001] spec_forced_reject vs target_only: diverge at pos 1 (len 256 vs 256); [r006] spec_forced_reject vs target_only: diverge at pos 28 (len 256 vs 256) |
| i2-natural-k1 | accept_path_equivalence | fail | [r000] spec_full_accept vs target_only: diverge at pos 1 (len 32 vs 32); [r003] spec_full_accept vs target_only: diverge at pos 0 (len 128 vs 128); [r004] spec_full_accept vs target_only: diverge at pos 1 (len 256 vs 256) … |
| i3-perm-k1-eq | permutation_invariance | fail | [r005] order0 vs order1: diverge at pos 0 (len 96 vs 96); [r006] order0 vs order1: diverge at pos 4 (len 96 vs 96) |
| i3-perm-k1-rag | permutation_invariance | pass |  |
| i4-split-k1-eq | split_merge_equivalence | fail | [r000] packed vs per_request: diverge at pos 0 (len 96 vs 96); [r003] packed vs per_request: diverge at pos 0 (len 96 vs 96); [r004] packed vs per_request: diverge at pos 3 (len 96 vs 96) |
| i4-split-k1-rag | split_merge_equivalence | fail | [r001] packed vs per_request: diverge at pos 4 (len 24 vs 24) |
| i5-isolation-k1 | state_isolation | fail | [r000] pair[r000+r002] vs alone:r000: diverge at pos 2 (len 96 vs 96); [r002] pair[r002+r000] vs alone:r002: diverge at pos 3 (len 96 vs 96); [r002] pair[r001+r002] vs alone:r002: diverge at pos 1 (len 96 vs 96) … |
| i6-graph-k1-n1 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n2 | graph_geometry_consistency | fail | [r000] captured vs target_only: diverge at pos 1 (len 64 vs 64) |
| i6-graph-k1-n3 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 9 (len 64 vs 64); [r000] captured vs target_only: diverge at pos 9 (len 64 vs 64) |
| i6-graph-k1-n4 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 3 (len 64 vs 64); [r001] captured vs eager: diverge at pos 1 (len 64 vs 64); [r000] captured vs target_only: diverge at pos 0 (len 64 vs 64) … |
| i6-graph-k1-n5 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 8 (len 64 vs 64); [r001] captured vs eager: diverge at pos 8 (len 64 vs 64); [r000] captured vs target_only: diverge at pos 8 (len 64 vs 64) … |
| i6-graph-k1-n7 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 0 (len 64 vs 64); [r001] captured vs eager: diverge at pos 3 (len 64 vs 64); [r005] captured vs eager: diverge at pos 1 (len 64 vs 64) … |
| i6-graph-k1-n8 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 1 (len 64 vs 64); [r007] captured vs eager: diverge at pos 0 (len 64 vs 64); [r000] captured vs target_only: diverge at pos 1 (len 64 vs 64) … |
| i6-graph-k1-n15 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 0 (len 64 vs 64); [r010] captured vs eager: diverge at pos 1 (len 64 vs 64); [r013] captured vs eager: diverge at pos 9 (len 64 vs 64) … |
| i6-graph-k1-n16 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 1 (len 64 vs 64); [r002] captured vs eager: diverge at pos 0 (len 64 vs 64); [r003] captured vs eager: diverge at pos 3 (len 64 vs 64) … |
| i1-forced-reject-k3-s | all_reject_equivalence | pass |  |
| i1-forced-reject-k3-l | all_reject_equivalence | fail | [r001] spec_forced_reject vs target_only: diverge at pos 0 (len 256 vs 256); [r002] spec_forced_reject vs target_only: diverge at pos 3 (len 256 vs 256); [r006] spec_forced_reject vs target_only: diverge at pos 0 (len 256 vs 256) |
| i2-natural-k3 | accept_path_equivalence | fail | [r003] spec_full_accept vs target_only: diverge at pos 2 (len 256 vs 256); [r004] spec_full_accept vs target_only: diverge at pos 0 (len 128 vs 128); [r007] spec_full_accept vs target_only: diverge at pos 0 (len 32 vs 32) … |
| i3-perm-k3-eq | permutation_invariance | fail | [r002] order0 vs order1: diverge at pos 0 (len 96 vs 96) |
| i3-perm-k3-rag | permutation_invariance | fail | [r001] order0 vs order1: diverge at pos 0 (len 192 vs 192); [r006] order0 vs order1: diverge at pos 0 (len 192 vs 192); [r007] order0 vs order1: diverge at pos 1 (len 48 vs 48) |
| i4-split-k3-eq | split_merge_equivalence | fail | [r000] packed vs per_request: diverge at pos 1 (len 96 vs 96); [r002] packed vs per_request: diverge at pos 1 (len 96 vs 96); [r007] packed vs per_request: diverge at pos 2 (len 96 vs 96) |
| i4-split-k3-rag | split_merge_equivalence | fail | [r002] packed vs per_request: diverge at pos 12 (len 192 vs 192); [r006] packed vs per_request: diverge at pos 5 (len 24 vs 24) |
| i5-isolation-k3 | state_isolation | fail | [r001] pair[r000+r001] vs alone:r001: diverge at pos 3 (len 96 vs 96); [r001] pair[r001+r000] vs alone:r001: diverge at pos 3 (len 96 vs 96); [r001] pair[r001+r002] vs alone:r001: diverge at pos 3 (len 96 vs 96) … |
| i6-graph-k3-n1 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 0 (len 64 vs 64); [r000] captured vs target_only: diverge at pos 0 (len 64 vs 64) |
| i6-graph-k3-n2 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n3 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 2 (len 64 vs 64); [r000] captured vs target_only: diverge at pos 2 (len 64 vs 64); [r001] captured vs target_only: diverge at pos 0 (len 64 vs 64) |
| i6-graph-k3-n4 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n5 | graph_geometry_consistency | fail | [r003] captured vs eager: diverge at pos 3 (len 64 vs 64); [r004] captured vs eager: diverge at pos 2 (len 64 vs 64) |
| i6-graph-k3-n7 | graph_geometry_consistency | fail | [r002] captured vs eager: diverge at pos 37 (len 64 vs 64); [r005] captured vs eager: diverge at pos 4 (len 64 vs 64); [r002] captured vs target_only: diverge at pos 37 (len 64 vs 64) … |
| i6-graph-k3-n8 | graph_geometry_consistency | fail | [r000] captured vs eager: diverge at pos 7 (len 64 vs 64); [r002] captured vs eager: diverge at pos 0 (len 64 vs 64); [r003] captured vs eager: diverge at pos 0 (len 64 vs 64) … |
| i6-graph-k3-n15 | graph_geometry_consistency | fail | [r007] captured vs eager: diverge at pos 18 (len 64 vs 64); [r008] captured vs eager: diverge at pos 0 (len 64 vs 64); [r009] captured vs eager: diverge at pos 0 (len 64 vs 64) … |
| i6-graph-k3-n16 | graph_geometry_consistency | fail | [r008] captured vs eager: diverge at pos 5 (len 64 vs 64); [r009] captured vs eager: diverge at pos 1 (len 64 vs 64); [r010] captured vs eager: diverge at pos 0 (len 64 vs 64) … |

