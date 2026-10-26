# SpecSafe run `20261026_121800-floor`

- matrix: **floor**
- engine: **vllm-offline**, margin tau=0.5, 1x repeats
- started: 2026-10-26T12:18:00  duration: 117.1s
- verdict: **ALL PASS** (2 pass / 0 boundary / 0 fail / 0 error)

Verdict legend: PASS = token-identical; BOUNDARY = divergence only at near-tie positions (margin <= tau, numeric floor); FAIL = confident divergence (margin > tau both sides).

## Engine

- vllm_version: `0.32.0`
- model: `/root/autodl-tmp/models/qwen3.8-27b-int4`
- max_model_len: `4096`
- tp: `1`
- capabilities: `('graph_regime', 'logprob_margin')`
- resolved_spec_configs: `{"(('attention_config', (('use_trtllm_attention', 0),)), ('gpu_memory_utilization', 0.85))": {'note': 'spec config echo unavailable (internals moved)'}}`

## Results

| case | invariant | status | detail |
|---|---|---|---|
| floor-rebuild-n3 | rebuild_floor | **pass** |  |
| floor-batch-n8 | batch_floor | **pass** |  |
