# SpecSafe run `20261024_125201-static32`

- matrix: **static32**
- engine: **reference**
- started: 2026-10-24T12:52:01  duration: 1.5s
- verdict: **ALL PASS** (34 pass / 0 fail / 0 error)

## Engine

- vocab: `101`
- impl: `pure-python`
- capabilities: `('draft_mode', 'state_digest', 'graph_regime')`

## Results

| case | invariant | status | detail |
|---|---|---|---|
| i1-forced-reject-k1-s | all_reject_equivalence | pass |  |
| i1-forced-reject-k1-l | all_reject_equivalence | pass |  |
| i2-natural-k1 | accept_path_equivalence | pass |  |
| i3-perm-k1-eq | permutation_invariance | pass |  |
| i3-perm-k1-rag | permutation_invariance | pass |  |
| i4-split-k1-eq | split_merge_equivalence | pass |  |
| i4-split-k1-rag | split_merge_equivalence | pass |  |
| i5-isolation-k1 | state_isolation | pass |  |
| i6-graph-k1-n1 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n2 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n3 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n4 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n5 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n7 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n8 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n15 | graph_geometry_consistency | pass |  |
| i6-graph-k1-n16 | graph_geometry_consistency | pass |  |
| i1-forced-reject-k3-s | all_reject_equivalence | pass |  |
| i1-forced-reject-k3-l | all_reject_equivalence | pass |  |
| i2-natural-k3 | accept_path_equivalence | pass |  |
| i3-perm-k3-eq | permutation_invariance | pass |  |
| i3-perm-k3-rag | permutation_invariance | pass |  |
| i4-split-k3-eq | split_merge_equivalence | pass |  |
| i4-split-k3-rag | split_merge_equivalence | pass |  |
| i5-isolation-k3 | state_isolation | pass |  |
| i6-graph-k3-n1 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n2 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n3 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n4 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n5 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n7 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n8 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n15 | graph_geometry_consistency | pass |  |
| i6-graph-k3-n16 | graph_geometry_consistency | pass |  |
