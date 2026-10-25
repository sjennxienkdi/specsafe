"""Each mutant must be caught by its invariant -- the framework's teeth.

v2: all catches happen under the margin-aware oracle (tau=0.5): mutants
diverge at confident positions (margin 3.0) and must NOT be washed to
BOUNDARY. See test_margin_oracle.py for the complementary honesty test
(NoisyFlip must be spared)."""
import pytest

from specsafe.engine.mutants import MUTANTS
from specsafe.engine.reference import ReferenceEngine
from specsafe.generator import build_matrix
from specsafe.invariants import JudgeConfig, run_case

_CASES = {c.invariant: c for c in build_matrix("selftest")}


@pytest.mark.parametrize("invariant,mutant_cls", sorted(MUTANTS.items()))
def test_mutant_is_caught_by_its_invariant(invariant, mutant_cls):
    out = run_case(mutant_cls(), _CASES[invariant], JudgeConfig(tau=0.5))
    assert out.status == "fail", (
        f"{mutant_cls.__name__} escaped {invariant}: status={out.status} "
        f"{out.details}")


@pytest.mark.parametrize("invariant,mutant_cls", sorted(MUTANTS.items()))
def test_mutant_caught_but_clean_passes_same_case(invariant, mutant_cls):
    clean = run_case(ReferenceEngine(), _CASES[invariant], JudgeConfig(tau=0.5))
    assert clean.status == "pass"


def test_reject_bonus_drop_is_exactly_the_35150_shape():
    """Forced-reject arm: output diverges from target-only and the FIRST
    divergence is not at position 0 (drift accumulates, not instant)."""
    from specsafe.engine.mutants import RejectBonusDrop
    from specsafe.invariants import check_all_reject_equivalence
    out = check_all_reject_equivalence(
        RejectBonusDrop(), _CASES["all_reject_equivalence"], JudgeConfig(tau=0.5))
    assert out.status == "fail"
    first = min((m.first_divergence for m in out.mismatches
                 if m.first_divergence is not None), default=None)
    assert first is None or first > 0, "drift class: divergence must accumulate"


def test_graph_mutant_only_corrupts_captured_regime():
    from specsafe.engine.mutants import GraphRegimeCorrupt
    from specsafe.invariants import check_graph_geometry_consistency
    out = check_graph_geometry_consistency(
        GraphRegimeCorrupt(), _CASES["graph_geometry_consistency"],
        JudgeConfig(tau=0.5))
    assert out.status == "fail"
    # the corruption is replay-specific: the captured arm is the one that
    # diverges from target-only
    assert any(m.arm_a == "captured" and m.arm_b == "target_only"
               for m in out.mismatches)
