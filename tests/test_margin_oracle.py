"""Oracle v2: three-tier verdicts, margin classification, repeat majority,
noise-double honesty, floor matrix."""
import pytest

from specsafe.engine.base import GenResult
from specsafe.engine.mutants import MUTANTS, NoisyFlip
from specsafe.engine.reference import ReferenceEngine
from specsafe.generator import build_matrix
from specsafe.invariants import (
    JudgeConfig,
    Mismatch,
    _aggregate,
    _compare,
    _verdict_of,
    run_arm,
    run_case,
)
from specsafe.scenario import EngineOptions, ReqSpec, Scenario


def _sc(n=3, mt=128, k=3):
    reqs = tuple(ReqSpec(f"r{i:03d}", f"margin-test-{i}", mt) for i in range(n))
    return Scenario("t", "t", reqs,
                    EngineOptions(method="spec", num_speculative_tokens=k))


def _res(rid, toks, margins):
    return GenResult(rid=rid, token_ids=list(toks),
                     margins=list(margins) if margins is not None else None)


# ---------------------------------------------------------------- classify

def test_classify_equal_is_pass():
    a = {"r0": _res("r0", [1, 2, 3], [3.0, 3.0, 3.0])}
    b = {"r0": _res("r0", [1, 2, 3], [3.0, 3.0, 3.0])}
    ms = _compare("a", a, "b", b, False, 0.5)
    assert ms == [] and _verdict_of(ms) == "pass"


def test_classify_near_tie_both_sides_is_boundary():
    a = {"r0": _res("r0", [1, 2, 9], [3.0, 3.0, 0.1])}
    b = {"r0": _res("r0", [1, 2, 7], [3.0, 3.0, 0.25])}
    ms = _compare("a", a, "b", b, False, 0.5)
    assert len(ms) == 1 and ms[0].verdict == "boundary"
    assert _verdict_of(ms) == "boundary"


def test_classify_one_sided_near_tie_is_boundary():
    # a near-tie in EITHER arm explains the flip
    a = {"r0": _res("r0", [1, 9], [3.0, 0.0])}
    b = {"r0": _res("r0", [1, 7], [3.0, 3.0])}
    ms = _compare("a", a, "b", b, False, 0.5)
    assert ms[0].verdict == "boundary"


def test_classify_confident_divergence_is_fail():
    a = {"r0": _res("r0", [1, 9], [3.0, 3.0])}
    b = {"r0": _res("r0", [1, 7], [3.0, 3.0])}
    ms = _compare("a", a, "b", b, False, 0.5)
    assert ms[0].verdict == "fail" and _verdict_of(ms) == "fail"


def test_classify_missing_margins_divergence_is_fail():
    # no margin evidence -> no boundary credit (conservative)
    a = {"r0": _res("r0", [1, 9], None)}
    b = {"r0": _res("r0", [1, 7], None)}
    ms = _compare("a", a, "b", b, False, 0.5)
    assert ms[0].verdict == "fail"


def test_classify_worst_wins_within_case():
    a = {"r0": _res("r0", [1, 9], [3.0, 3.0]),
         "r1": _res("r1", [5, 4], [0.0, 3.0])}
    b = {"r0": _res("r0", [1, 7], [3.0, 3.0]),
         "r1": _res("r1", [5, 3], [0.1, 3.0])}
    ms = _compare("a", a, "b", b, False, 0.5)
    assert _verdict_of(ms) == "fail"  # r0 fail dominates r1 boundary


# ---------------------------------------------------------------- aggregate

def _outcome(status):
    from specsafe.invariants import InvariantOutcome
    return InvariantOutcome("x", "c", "s", status)


def test_aggregate_majority_fail():
    assert _aggregate([_outcome("fail"), _outcome("fail"), _outcome("pass")]).status == "fail"


def test_aggregate_majority_boundary():
    r = _aggregate([_outcome("boundary"), _outcome("boundary"), _outcome("pass")])
    assert r.status == "boundary"


def test_aggregate_mixed_noise_not_pass():
    # 1 fail + 1 boundary + 1 pass: majority is "divergent" -> boundary
    r = _aggregate([_outcome("fail"), _outcome("boundary"), _outcome("pass")])
    assert r.status == "boundary"


def test_aggregate_all_pass():
    r = _aggregate([_outcome("pass")] * 3)
    assert r.status == "pass" and r.repeat_verdicts == ["pass"] * 3


def test_aggregate_majority_error():
    r = _aggregate([_outcome("error"), _outcome("error"), _outcome("pass")])
    assert r.status == "error"


def test_aggregate_minority_error_excluded():
    r = _aggregate([_outcome("pass"), _outcome("error"), _outcome("pass")])
    assert r.status == "pass" and "errored" in r.details


# ---------------------------------------------------------------- engine doubles

def test_reference_margins_deterministic_and_shaped():
    eng, sc = ReferenceEngine(), _sc(mt=96)
    a = eng.generate(sc.requests, EngineOptions(method="none"))
    b = eng.generate(sc.requests, EngineOptions(method="none"))
    for ra, rb in zip(a, b):
        assert ra.margins == rb.margins
        assert len(ra.margins) == len(ra.token_ids)
        assert all(m == 3.0 for m in ra.margins[:32])      # early confident
        assert any(m < 0.5 for m in ra.margins[32:])       # later near-ties


def test_noisy_flip_diverges_only_at_near_ties_and_oracle_spared():
    sc = _sc(n=3, mt=128)
    opts = EngineOptions(method="none", seed=99)
    a = run_arm(ReferenceEngine(), sc, opts, label="clean")
    b = run_arm(NoisyFlip(), sc, opts, label="noisy")
    m05 = _compare("clean", a, "noisy", b, False, 0.5)
    assert m05, "noise double must actually diverge in this window"
    assert all(m.verdict == "boundary" for m in m05)
    assert _verdict_of(m05) == "boundary"
    # strict oracle (tau=0) must NOT spare it -- the threshold is doing work
    m00 = _compare("clean", a, "noisy", b, False, 0.0)
    assert _verdict_of(m00) == "fail"


@pytest.mark.parametrize("invariant,mutant_cls", sorted(MUTANTS.items()))
def test_mutants_still_fail_under_margin_oracle(invariant, mutant_cls):
    case = next(c for c in build_matrix("selftest") if c.invariant == invariant)
    out = run_case(mutant_cls(), case, JudgeConfig(tau=0.5))
    assert out.status == "fail", (
        f"{mutant_cls.__name__} washed to {out.status} by margin oracle: "
        + "; ".join(m.render() for m in out.mismatches[:3]))


def test_clean_engine_passes_with_repeats():
    eng = ReferenceEngine()
    for case in build_matrix("selftest"):
        out = run_case(eng, case, JudgeConfig(tau=0.5, repeats=3))
        assert out.status == "pass", (case.case_id, out.repeat_verdicts)


# ---------------------------------------------------------------- floor

def test_floor_matrix_reference_is_flat():
    """Deterministic reference engine: no rebuild or batch floor measurable;
    both floor cases must come out pass."""
    eng = ReferenceEngine()
    for case in build_matrix("floor"):
        out = run_case(eng, case, JudgeConfig(tau=0.5))
        assert out.status == "pass", (case.case_id, out.details)
        assert "floor not measurable" in out.details


def test_floor_rebuild_detects_noisy_rebuilds():
    """If rebuilds were noisy (simulate by flipping the engine between
    builds), rebuild_floor must surface divergences with margin stats."""
    from specsafe.floor import check_rebuild_floor
    from specsafe.invariants import JudgeConfig

    class AlternatingEngine:
        """Clean build for b0/b2, NoisyFlip for b1 -- mimics noisy rebuild."""
        name = "alt"
        _flip = False

        def generate(self, requests, opts):
            tags = dict(opts.extra).get("probe_build", "b0")
            eng = NoisyFlip() if tags == "b1" else ReferenceEngine()
            return eng.generate(requests, opts)

        def info(self):
            from specsafe.engine.base import EngineInfo
            return EngineInfo(name=self.name, details={})

    case = build_matrix("floor")[0]
    out = check_rebuild_floor(AlternatingEngine(), case, JudgeConfig(tau=0.5))
    assert out.status == "boundary"          # flips only at near-tie positions
    assert "suggested_tau" in out.details    # calibration data attached
