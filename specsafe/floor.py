"""Non-determinism floor measurement (the probe_pos0 capability, in-tool).

Two families:

  rebuild_floor  -- run the SAME config through N independent engine builds
                    and compare pairwise. Measures the build-noise floor:
                    divergence rate + margin distribution at divergences.
                    On real GPUs this is the A-vs-A' experiment that showed
                    vLLM 0.32.0 greedy flips near-ties across builds
                    (results/gpu-20261024/probe_pos0.json). Engines rebuild
                    when the scenario carries distinct probe_build tags in
                    opts.extra ("probe_build", "<tag>") -- the vLLM adapter
                    keys engines on config and treats probe_* extras as
                    metadata, forcing a fresh LLM per tag.

  batch_floor    -- same requests, target-only, different batch compositions
                    (packed vs per-request vs shuffled orders) within ONE
                    build. Divergence here localizes the noise to
                    batch-composition-dependent kernel paths (vs rebuild-only
                    noise); the margin split (0.0 exact ties vs small gaps)
                    separates tie-break flips from accumulation differences.

Both report per-divergence margins so tau can be calibrated from data:
suggested tau = max observed boundary-class margin + headroom.
"""
from __future__ import annotations

from .engine.base import Engine, GenResult
from .invariants import (
    ArmSummary,
    InvariantOutcome,
    JudgeConfig,
    _compare,
    _first_divergence,
    _margin_at,
    _summary,
    _target_only,
    _verdict_of,
    run_arm,
)
from .scenario import CaseSpec


def check_rebuild_floor(engine: Engine, case: CaseSpec,
                        judge: JudgeConfig) -> InvariantOutcome:
    """Pairwise comparison across N independent engine builds (same config).

    Scenario contract: spec_opts.extra carries ("probe_builds", str(n));
    builds are forced via ("probe_build", f"b{i}") tags."""
    sc = case.scenario
    opts = sc.spec_opts
    o = InvariantOutcome("rebuild_floor", case.case_id, sc.scenario_id,
                         margin_tau=judge.tau)
    n_builds = 2
    for k, v in opts.extra:
        if k == "probe_builds":
            n_builds = max(2, int(v))
    arms: dict[str, dict[str, GenResult]] = {}
    for i in range(n_builds):
        build_opts = opts.with_(extra=opts.extra + (("probe_build", f"b{i}"),))
        arms[f"build{i}"] = run_arm(engine, sc, build_opts,
                                    label=f"build{i}")
        o.arms.append(_summary(f"build{i}", arms[f"build{i}"]))
    keys = list(arms)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            o.mismatches += _compare(keys[i], arms[keys[i]], keys[j],
                                     arms[keys[j]], False, judge.tau)
    o.status = _verdict_of(o.mismatches)
    _annotate_floor_stats(o, judge.tau)
    return o


def check_batch_floor(engine: Engine, case: CaseSpec,
                      judge: JudgeConfig) -> InvariantOutcome:
    """Same requests, one build, different batch compositions: packed vs
    per-request vs permutations. Divergence here = batch-shape-dependent
    numerics (kernel tiling / padding), not build noise."""
    sc = case.scenario
    opts = _target_only(sc.spec_opts)
    o = InvariantOutcome("batch_floor", case.case_id, sc.scenario_id,
                         margin_tau=judge.tau)
    packed = run_arm(engine, sc, opts, label="packed")
    singles = run_arm(engine, sc, opts, single=True, label="per_request")
    o.arms = [_summary("packed", packed), _summary("per_request", singles)]
    o.mismatches += _compare("packed", packed, "per_request", singles,
                             False, judge.tau)
    n = len(sc.requests)
    orders = [tuple(range(n))]
    orders += [tuple(p) for p in sc.permutations if len(p) == n]
    if not sc.permutations:
        orders.append(tuple(reversed(range(n))))
    for i, order in enumerate(orders[1:], 1):
        run = run_arm(engine, sc, opts, order=order, label=f"order{i}")
        o.arms.append(_summary(f"order{i}", run))
        o.mismatches += _compare("packed", packed, f"order{i}", run,
                                 False, judge.tau)
    o.status = _verdict_of(o.mismatches)
    _annotate_floor_stats(o, judge.tau)
    return o


def _annotate_floor_stats(o: InvariantOutcome, tau: float) -> None:
    """Attach divergence/margin statistics to details for tau calibration."""
    if not o.mismatches:
        o.details = "no divergence across builds/compositions (floor not measurable here)"
        return
    margins = [m for ms in ([mm.margin_a, mm.margin_b]
                            for mm in o.mismatches) for m in ms
               if m is not None]
    n_pair = max(1, len(o.arms))
    n_div = len(o.mismatches)
    if margins:
        ties = sum(1 for m in margins if m <= 0.0)
        lo, hi = min(margins), max(margins)
        stats = (f"divergences={n_div} margins[min={lo} max={hi} "
                 f"exact_ties={ties}/{len(margins)}] "
                 f"suggested_tau>{hi}")
    else:
        stats = f"divergences={n_div} (no margin evidence available)"
    o.details = f"floor: {stats}; tau={tau}"


FLOOR_CHECKERS = {
    "rebuild_floor": check_rebuild_floor,
    "batch_floor": check_batch_floor,
}

# late registration: floor.py imports invariants, so invariants cannot
# import floor at module level
from .invariants import CHECKERS as _CHECKERS  # noqa: E402
_CHECKERS.update(FLOOR_CHECKERS)
