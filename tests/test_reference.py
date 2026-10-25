"""Reference engine semantics: the toy model must behave like a *correct*
spec-decode engine before we use it to judge real ones."""
import pytest

from specsafe.engine.reference import ReferenceEngine, fnv1a
from specsafe.generator import build_matrix
from specsafe.invariants import run_case
from specsafe.scenario import (
    EngineOptions,
    ReqSpec,
    Scenario,
    unconditional_to_conditional,
    validate_rates,
)


def _sc(n=3, mt=40, k=3):
    reqs = tuple(ReqSpec(f"r{i}", f"prompt-{i}", mt) for i in range(n))
    return Scenario("t", "t", reqs, EngineOptions(method="spec",
                                                  num_speculative_tokens=k))


def _tokens(res):
    return list(res.token_ids)


def test_deterministic_across_calls():
    eng, sc = ReferenceEngine(), _sc()
    a = eng.generate(sc.requests, sc.spec_opts)
    b = eng.generate(sc.requests, sc.spec_opts)
    assert [_tokens(r) for r in a] == [_tokens(r) for r in b]


def test_standard_oracle_equals_target_only():
    """Lossless verification: perfect drafter, all accepted -> identical to
    target-only greedy. This is the all-accept invariant in miniature."""
    eng, sc = ReferenceEngine(), _sc()
    spec = eng.generate(sc.requests, sc.spec_opts.with_(
        rejection_sample_method="standard", draft_mode="oracle"))
    base = eng.generate(sc.requests, EngineOptions(method="none"))
    for s, b in zip(spec, base):
        assert _tokens(s) == _tokens(b)
        assert all(a == 4 for a in s.accept_lens)  # K=3 drafts + bonus


def test_standard_adversarial_forces_all_reject_and_stays_lossless():
    """Every draft rejected -> output still equals target-only (SGLang #35150
    property) and every step advances exactly one token."""
    eng, sc = ReferenceEngine(), _sc()
    spec = eng.generate(sc.requests, sc.spec_opts.with_(
        rejection_sample_method="standard", draft_mode="adversarial"))
    base = eng.generate(sc.requests, EngineOptions(method="none"))
    for s, b in zip(spec, base):
        assert _tokens(s) == _tokens(b)
        assert all(a == 1 for a in s.accept_lens)
        assert s.num_steps == len(b.token_ids)


def test_synthetic_zero_rates_equal_target_only():
    eng, sc = ReferenceEngine(), _sc()
    spec = eng.generate(sc.requests, sc.spec_opts.with_(
        rejection_sample_method="synthetic",
        synthetic_acceptance_rates=(0.0, 0.0, 0.0)))
    base = eng.generate(sc.requests, EngineOptions(method="none"))
    for s, b in zip(spec, base):
        assert _tokens(s) == _tokens(b)


def test_synthetic_full_rates_accept_everything():
    eng, sc = ReferenceEngine(), _sc()
    spec = eng.generate(sc.requests, sc.spec_opts.with_(
        rejection_sample_method="synthetic",
        synthetic_acceptance_rates=(1.0, 1.0, 1.0)))
    for s in spec:
        assert all(a == 4 for a in s.accept_lens)
        # K+1 tokens per step: far fewer steps than tokens
        assert s.num_steps == (len(s.token_ids) + 3) // 4


def test_length_never_exceeds_max_tokens():
    eng, sc = ReferenceEngine(), _sc(mt=17)
    for mode in ("oracle", "learned"):
        res = eng.generate(sc.requests, sc.spec_opts.with_(
            rejection_sample_method="standard", draft_mode=mode))
        for r in res:
            assert len(r.token_ids) == 17


def test_mixed_acceptance_occurs_naturally():
    """The learned drafter (~2/3) must produce BOTH accept and reject events,
    otherwise accept-path bugs would be invisible to the natural arm."""
    eng, sc = ReferenceEngine(), _sc(n=4, mt=80)
    res = eng.generate(sc.requests, sc.spec_opts.with_(
        rejection_sample_method="standard"))
    lens = [a for r in res for a in r.accept_lens]
    assert any(a == 4 for a in lens) and any(a < 4 for a in lens)


def test_rates_validation():
    with pytest.raises(ValueError):
        validate_rates((0.5,), k=3)
    with pytest.raises(ValueError):
        validate_rates((0.2, 0.5, 0.5), k=3)   # increasing
    with pytest.raises(ValueError):
        validate_rates((0.5, -0.1, 0.0), k=3)
    validate_rates((1.0, 0.5, 0.0), k=3)


def test_unconditional_to_conditional():
    assert unconditional_to_conditional((0.0, 0.0)) == (0.0, 0.0)
    assert unconditional_to_conditional((1.0, 1.0)) == (1.0, 1.0)
    assert unconditional_to_conditional((0.5, 0.25)) == (0.5, 0.5)


def test_fnv_stable():
    assert fnv1a([1, 2, 3]) == fnv1a([1, 2, 3]) != fnv1a([3, 2, 1])


def test_clean_engine_passes_all_six():
    eng = ReferenceEngine()
    for case in build_matrix("selftest"):
        out = run_case(eng, case)
        assert out.status == "pass", (case.case_id, [m.render() for m in out.mismatches])
