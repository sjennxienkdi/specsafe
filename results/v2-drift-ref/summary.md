# SpecSafe run `20261024_201228-drift35150`

- matrix: **drift35150**
- engine: **reference**, margin tau=0.5, 1x repeats
- started: 2026-10-24T20:12:28  duration: 0.7s
- verdict: **ALL PASS** (3 pass / 0 boundary / 0 fail / 0 error)

Verdict legend: PASS = token-identical; BOUNDARY = divergence only at near-tie positions (margin <= tau, numeric floor); FAIL = confident divergence (margin > tau both sides).

## Engine

- vocab: `101`
- impl: `pure-python`
- capabilities: `('draft_mode', 'state_digest', 'graph_regime')`

## Results

| case | invariant | status | detail |
|---|---|---|---|
| drift-k1-n4-L256 | all_reject_equivalence | **pass** |  |
| drift-k3-n8-L512 | all_reject_equivalence | **pass** |  |
| drift-k7-n32-L256 | all_reject_equivalence | **pass** |  |
