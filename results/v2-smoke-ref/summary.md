# SpecSafe run `20261024_201227-smoke`

- matrix: **smoke**
- engine: **reference**, margin tau=0.5, 3x repeats
- started: 2026-10-24T20:12:27  duration: 0.7s
- verdict: **ALL PASS** (12 pass / 0 boundary / 0 fail / 0 error)

Verdict legend: PASS = token-identical; BOUNDARY = divergence only at near-tie positions (margin <= tau, numeric floor); FAIL = confident divergence (margin > tau both sides).

## Engine

- vocab: `101`
- impl: `pure-python`
- capabilities: `('draft_mode', 'state_digest', 'graph_regime')`

## Results

| case | invariant | status | detail |
|---|---|---|---|
| smoke-k1-all_reject_equivalence | all_reject_equivalence | **pass** | [repeats pass,pass,pass]  |
| smoke-k1-accept_path_equivalence | accept_path_equivalence | **pass** | [repeats pass,pass,pass]  |
| smoke-k1-permutation_invariance | permutation_invariance | **pass** | [repeats pass,pass,pass]  |
| smoke-k1-split_merge_equivalence | split_merge_equivalence | **pass** | [repeats pass,pass,pass]  |
| smoke-k1-state_isolation | state_isolation | **pass** | [repeats pass,pass,pass]  |
| smoke-k1-graph_geometry_consistency | graph_geometry_consistency | **pass** | [repeats pass,pass,pass]  |
| smoke-k3-all_reject_equivalence | all_reject_equivalence | **pass** | [repeats pass,pass,pass]  |
| smoke-k3-accept_path_equivalence | accept_path_equivalence | **pass** | [repeats pass,pass,pass]  |
| smoke-k3-permutation_invariance | permutation_invariance | **pass** | [repeats pass,pass,pass]  |
| smoke-k3-split_merge_equivalence | split_merge_equivalence | **pass** | [repeats pass,pass,pass]  |
| smoke-k3-state_isolation | state_isolation | **pass** | [repeats pass,pass,pass]  |
| smoke-k3-graph_geometry_consistency | graph_geometry_consistency | **pass** | [repeats pass,pass,pass]  |
