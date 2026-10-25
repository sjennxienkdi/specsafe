"""Deterministic scenario generation.

The generator turns dimension axes (acceptance schedules, batch shapes,
lengths, seeds) into Scenario objects. Everything is seeded and stable: the
same matrix name always produces byte-identical scenario JSON, so a scenario
generated locally is exactly what the cloud replays.
"""
from __future__ import annotations

import random

from .scenario import CaseSpec, EngineOptions, ReqSpec, Scenario

_PREFIX = "specsafe-probe"


def _rng(matrix: str, seed: int) -> random.Random:
    return random.Random(f"{_PREFIX}:{matrix}:{seed}")


def _prompt(rng: random.Random, idx: int, flavor: str = "") -> str:
    """Prompt text. Content is irrelevant to correctness invariants -- only
    identity across arms matters. `ragged` flavors vary lengths deliberately."""
    words = [f"{_PREFIX}-{flavor}-{idx}", f"seed-{rng.randrange(1 << 30):08x}"]
    n_fill = rng.randrange(1, 24)
    words += [rng.choice(["alpha", "beta", "gamma", "delta", "epsilon",
                          "zeta", "eta", "theta", "iota", "kappa"])
              for _ in range(n_fill)]
    return " ".join(words)


def make_requests(rng: random.Random, n: int, max_tokens: int,
                  flavor: str = "", ragged: bool = False) -> tuple[ReqSpec, ...]:
    reqs = []
    for i in range(n):
        mt = max_tokens
        if ragged and n > 1:
            # mixed lengths spanning bucket boundaries: 1/4x .. 2x
            mt = max(8, int(max_tokens * rng.choice([0.25, 0.5, 1.0, 1.0, 2.0])))
        reqs.append(ReqSpec(f"r{i:03d}", _prompt(rng, i, flavor), mt))
    return tuple(reqs)


def spec_opts(k: int, seed: int = 0, method: str = "spec",
              rates: tuple[float, ...] | None = None) -> EngineOptions:
    opts = EngineOptions(method=method, num_speculative_tokens=k, seed=seed)
    if rates is not None:
        opts = opts.with_(rejection_sample_method="synthetic",
                          synthetic_acceptance_rates=rates)
    return opts


def base_scenario(sid: str, desc: str, reqs: tuple[ReqSpec, ...],
                  opts: EngineOptions, seed: int = 0,
                  permutations: tuple[tuple[int, ...], ...] = ()) -> Scenario:
    return Scenario(sid, desc, reqs, opts, permutations, seed)


# ------------------------------------------------------------------ matrices

def matrix_selftest() -> list[CaseSpec]:
    """One small case per invariant; used by `specsafe selftest` to prove the
    checkers catch each mutant on the reference engine."""
    rng = _rng("selftest", 7)
    reqs = make_requests(rng, 3, 40, "selftest", ragged=False)
    k = 3
    perm = (tuple(reversed(range(3))), (1, 2, 0))
    cases = []
    for inv in ("all_reject_equivalence", "accept_path_equivalence",
                "permutation_invariance", "split_merge_equivalence",
                "state_isolation", "graph_geometry_consistency"):
        sc = base_scenario(f"selftest/{inv}", f"selftest arm for {inv}",
                           reqs, spec_opts(k, seed=7), seed=7, permutations=perm)
        cases.append(CaseSpec(f"selftest-{inv}", inv, sc))
    return cases


def matrix_smoke() -> list[CaseSpec]:
    """Fast end-to-end pass over all six invariants; runnable on any engine
    (reference locally, any real model on the cloud box in minutes)."""
    cases: list[CaseSpec] = []
    for k in (1, 3):
        rng = _rng("smoke", 100 + k)
        n = 4
        reqs = make_requests(rng, n, 48, "smoke", ragged=(k == 3))
        perm = (tuple(reversed(range(n))), (2, 0, 3, 1))
        for inv in ("all_reject_equivalence", "accept_path_equivalence",
                    "permutation_invariance", "split_merge_equivalence",
                    "state_isolation", "graph_geometry_consistency"):
            sc = base_scenario(f"smoke/k{k}/{inv}", f"smoke k={k} {inv}",
                               reqs, spec_opts(k, seed=100 + k), seed=100 + k,
                               permutations=perm)
            cases.append(CaseSpec(f"smoke-k{k}-{inv}", inv, sc))
    return cases


def matrix_static32() -> list[CaseSpec]:
    """Cloud matrix for vLLM v0.32.0 static-K paths (methods mtp1/mtp3).

    Designed for a small fast model (e.g. Qwen3.5-0.8B + native MTP, the
    #53929 e2e pair): forced-reject, natural, permutation, split/merge,
    isolation, graph regime, plus bucket-boundary batch sizes for geometry.
    """
    cases: list[CaseSpec] = []

    def add(cid: str, inv: str, reqs: tuple[ReqSpec, ...], opts: EngineOptions,
            seed: int, perms: tuple[tuple[int, ...], ...] = ()) -> None:
        sc = base_scenario(f"static32/{cid}", cid, reqs, opts, seed, perms)
        cases.append(CaseSpec(cid, inv, sc))

    for k in (1, 3):
        # I1 forced reject, short + long
        for n, mt, tag in ((4, 64, "s"), (8, 256, "l")):
            rng = _rng("static32", 1000 * k + n)
            add(f"i1-forced-reject-k{k}-{tag}", "all_reject_equivalence",
                make_requests(rng, n, mt, "i1"), spec_opts(k, seed=k), k,
                ())
        # I2 natural lossless
        rng = _rng("static32", 2000 + k)
        add(f"i2-natural-k{k}", "accept_path_equivalence",
            make_requests(rng, 8, 128, "i2", ragged=True), spec_opts(k, seed=k), k)
        # I3 permutation: equal + ragged batches
        for ragged, tag in ((False, "eq"), (True, "rag")):
            rng = _rng("static32", 3000 + k + ragged)
            n = 8
            pm = tuple(tuple(_rng("static32", 77 + k + ragged).sample(range(n), n))
                       for _ in range(2))
            add(f"i3-perm-k{k}-{tag}", "permutation_invariance",
                make_requests(rng, n, 96, "i3", ragged=ragged),
                spec_opts(k, seed=k), k, pm)
        # I4 split/merge
        for ragged, tag in ((False, "eq"), (True, "rag")):
            rng = _rng("static32", 4000 + k + ragged)
            add(f"i4-split-k{k}-{tag}", "split_merge_equivalence",
                make_requests(rng, 8, 96, "i4", ragged=ragged),
                spec_opts(k, seed=k), k)
        # I5 isolation
        rng = _rng("static32", 5000 + k)
        add(f"i5-isolation-k{k}", "state_isolation",
            make_requests(rng, 4, 96, "i5"), spec_opts(k, seed=k), k)
        # I6 graph regime across bucket-boundary batch sizes
        for n in (1, 2, 3, 4, 5, 7, 8, 15, 16):
            rng = _rng("static32", 6000 + k * 100 + n)
            add(f"i6-graph-k{k}-n{n}", "graph_geometry_consistency",
                make_requests(rng, n, 64, "i6"), spec_opts(k, seed=k), k)
    return cases


def matrix_drift35150() -> list[CaseSpec]:
    """SGLang #35150 reproduction shape, applied to whatever engine is under
    test: forced-reject long generations, accumulated state over hundreds of
    steps, various batch shapes. Failure here = non-lossless forced reject."""
    cases: list[CaseSpec] = []
    for k, n, mt in ((1, 4, 256), (3, 8, 512), (7, 32, 256)):
        rng = _rng("drift35150", 9000 + k)
        reqs = make_requests(rng, n, mt, "drift")
        sc = base_scenario(f"drift35150/k{k}-n{n}-L{mt}",
                           f"#35150 shape: forced-reject drift k={k} n={n} L={mt}",
                           reqs, spec_opts(k, seed=k), seed=k)
        cases.append(CaseSpec(f"drift-k{k}-n{n}-L{mt}",
                              "all_reject_equivalence", sc))
    return cases


def matrix_floor() -> list[CaseSpec]:
    """Non-determinism floor measurement (Phase 2 of the v2 plan).

    rebuild_floor: N independent engine builds of the SAME config compared
    pairwise (on vLLM each probe_build tag forces a fresh LLM); measures
    the build-noise floor and calibrates tau from the margin distribution.
    batch_floor: one build, same requests, different batch compositions
    (packed vs per-request vs permutations) -- localizes noise to
    batch-shape-dependent kernel paths vs rebuild-only noise.
    """
    cases: list[CaseSpec] = []

    rng = _rng("floor", 42)
    reqs = make_requests(rng, 4, 64, "floor-rebuild")
    opts = EngineOptions(
        method="none", seed=42,
        extra=(("probe_builds", "3"),))
    cases.append(CaseSpec(
        "floor-rebuild-n3", "rebuild_floor",
        base_scenario("floor/rebuild-n3",
                      "3x independent rebuilds of the same config",
                      reqs, opts, seed=42)))

    rng = _rng("floor", 43)
    n = 8
    reqs = make_requests(rng, n, 96, "floor-batch")
    pm = (tuple(reversed(range(n))),
          tuple(_rng("floor", 77).sample(range(n), n)))
    cases.append(CaseSpec(
        "floor-batch-n8", "batch_floor",
        base_scenario("floor/batch-n8",
                      "packed vs per-request vs permutations, one build",
                      reqs, EngineOptions(method="none", seed=43),
                      seed=43, permutations=pm)))
    return cases


MATRICES = {
    "selftest": matrix_selftest,
    "smoke": matrix_smoke,
    "static32": matrix_static32,
    "drift35150": matrix_drift35150,
    "floor": matrix_floor,
}


def build_matrix(name: str) -> list[CaseSpec]:
    if name not in MATRICES:
        raise KeyError(f"unknown matrix '{name}'; available: {sorted(MATRICES)}")
    return MATRICES[name]()
