# SpecSafe run `20261024_201228-floor`

- matrix: **floor**
- engine: **reference**, margin tau=0.5, 1x repeats
- started: 2026-10-24T20:12:28  duration: 0.1s
- verdict: **ALL PASS** (2 pass / 0 boundary / 0 fail / 0 error)

Verdict legend: PASS = token-identical; BOUNDARY = divergence only at near-tie positions (margin <= tau, numeric floor); FAIL = confident divergence (margin > tau both sides).

## Engine

- vocab: `101`
- impl: `pure-python`
- capabilities: `('draft_mode', 'state_digest', 'graph_regime')`

## Results

| case | invariant | status | detail |
|---|---|---|---|
| floor-rebuild-n3 | rebuild_floor | **pass** |  |
| floor-batch-n8 | batch_floor | **pass** |  |
