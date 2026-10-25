"""SpecSafe: differential correctness testing for speculative decoding.

Target domain: hybrid GDN/Attention architectures (Qwen3.5 / Qwen3.8 /
Qwen3-Next family) running speculative decoding in vLLM / SGLang.

Six behavioral invariants (see specsafe.invariants):
  1. all_reject_equivalence   -- every draft rejected <=> target-only decode
  2. accept_path_equivalence  -- accepted prefix <=> target run of that prefix
  3. permutation_invariance   -- batch reordering leaves per-request outputs unchanged
  4. split_merge_equivalence  -- packed execution <=> per-request execution
  5. state_isolation          -- request A's accept/reject never changes B's output/state
  6. graph_geometry_consistency -- CUDA-graph regime changes leave outputs unchanged

Real-world bug classes these target:
  - SGLang #35150  (forced-reject GDN state drift)             -> invariant 1
  - vLLM   #53051  (prefill misdispatched into spec cudagraph,
                    silent GDN state loss)                      -> invariants 4/6
  - vLLM   #53929  (captured uniform layout vs ragged replay)   -> invariant 6
"""

__version__ = "0.2.0"
