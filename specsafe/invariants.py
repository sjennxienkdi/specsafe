"""The six invariant checkers, margin-aware oracle v2.

Verdict semantics (three-tier, per case):

  PASS      -- token streams identical across arms
  BOUNDARY  -- divergence(s) exist, but every divergence sits at a position
               where at least one arm's top1-top2 logprob margin <= tau
               (numeric near-tie; explainable by the build/batch noise floor
               measured on real GPUs -- see results/gpu-20261024/)
  FAIL      -- at least one divergence where BOTH arms were confident
               (margin > tau on both sides): a semantic difference the
               noise floor cannot explain

Each case may be repeated `repeats` times (fresh arm runs per repeat);
the case verdict is the majority with worst-wins ties (fail > boundary >
pass). Real bugs are deterministic enough to fail every repeat; noise
flips appear in only a minority of repeats.

Engines without margin support (margins=None on divergence) get no
boundary credit: divergence without evidence is FAIL (conservative).
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .engine.base import Engine, GenResult
from .scenario import CaseSpec, EngineOptions, Scenario

CAP_DRAFT_MODE = "draft_mode"          # engine can force oracle/adversarial drafts
CAP_STATE_DIGEST = "state_digest"      # engine exposes recurrent-state digests

# ---------------------------------------------------------------- judge

@dataclass(frozen=True)
class JudgeConfig:
    """Oracle configuration. tau is in nats (top1-top2 logprob gap)."""

    tau: float = 0.5
    repeats: int = 1

    def __post_init__(self) -> None:
        if self.tau < 0:
            raise ValueError("tau must be >= 0")
        if self.repeats < 1:
            raise ValueError("repeats must be >= 1")


# ---------------------------------------------------------------- results

@dataclass
class Mismatch:
    arm_a: str
    arm_b: str
    rid: str
    first_divergence: int | None       # token index, None = length-only diff
    len_a: int
    len_b: int
    state_a: int | None = None
    state_b: int | None = None
    margin_a: float | None = None      # top1-top2 gap of arm A at the divergence
    margin_b: float | None = None
    verdict: str = "fail"              # "fail" | "boundary"

    def render(self) -> str:
        at = "end" if self.first_divergence is None else f"pos {self.first_divergence}"
        s = ""
        if self.state_a is not None and self.state_b is not None and self.state_a != self.state_b:
            s = f" state {self.state_a:08x}!={self.state_b:08x}"
        mg = ""
        if self.margin_a is not None or self.margin_b is not None:
            mg = (f" margin a={self.margin_a} b={self.margin_b}"
                  f" -> {self.verdict.upper()}")
        return (f"[{self.rid}] {self.arm_a} vs {self.arm_b}: diverge at {at} "
                f"(len {self.len_a} vs {self.len_b}){s}{mg}")


@dataclass
class ArmSummary:
    label: str
    n_requests: int
    total_tokens: int
    mean_accept_len: float | None


@dataclass
class InvariantOutcome:
    invariant: str
    case_id: str
    scenario_id: str
    status: str = "pass"   # pass | boundary | fail | error (aggregated over repeats)
    details: str = ""
    mismatches: list[Mismatch] = field(default_factory=list)
    arms: list[ArmSummary] = field(default_factory=list)
    margin_tau: float | None = None
    repeat_verdicts: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status in ("pass", "boundary")


# ---------------------------------------------------------------- helpers

def _caps(engine: Engine) -> set[str]:
    return set(engine.info().details.get("capabilities", ()))


def _target_only(opts: EngineOptions) -> EngineOptions:
    return EngineOptions(method="none", seed=opts.seed,
                         graph_regime=opts.graph_regime)


def _ordered(reqs: Sequence, order: Sequence[int] | None) -> list:
    return list(reqs) if order is None else [reqs[i] for i in order]


def run_arm(engine: Engine, sc: Scenario, opts: EngineOptions,
            order: Sequence[int] | None = None, single: bool = False,
            label: str = "") -> dict[str, GenResult]:
    """Run the scenario's requests under opts. single=True executes each
    request as its own generate() call (independent execution)."""
    reqs = _ordered(sc.requests, order)
    if single:
        out: dict[str, GenResult] = {}
        for r in reqs:
            for res in engine.generate([r], opts):
                out[res.rid] = res
    else:
        out = {res.rid: res for res in engine.generate(reqs, opts)}
    return out


def _summary(label: str, results: dict[str, GenResult]) -> ArmSummary:
    acc = [a for r in results.values() if r.accept_lens for a in r.accept_lens]
    return ArmSummary(
        label=label,
        n_requests=len(results),
        total_tokens=sum(len(r.token_ids) for r in results.values()),
        mean_accept_len=round(sum(acc) / len(acc), 3) if acc else None,
    )


def _first_divergence(a: Sequence[int], b: Sequence[int]) -> int | None:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def _margin_at(res: GenResult, pos: int) -> float | None:
    if res.margins is not None and 0 <= pos < len(res.margins):
        return res.margins[pos]
    return None


def _classify(ra: GenResult, rb: GenResult, div: int | None, tau: float) -> str:
    """boundary iff near-tie evidence (margin <= tau) on at least one arm at
    the divergence position; otherwise fail (including no-evidence case)."""
    pos = div if div is not None else min(len(ra.token_ids), len(rb.token_ids))
    ma, mb = _margin_at(ra, pos), _margin_at(rb, pos)
    if (ma is not None and ma <= tau) or (mb is not None and mb <= tau):
        return "boundary"
    return "fail"


def _compare(label_a: str, arm_a: dict[str, GenResult],
             label_b: str, arm_b: dict[str, GenResult],
             check_state: bool, tau: float) -> list[Mismatch]:
    out = []
    for rid in sorted(arm_a):
        ra, rb = arm_a[rid], arm_b[rid]
        div = _first_divergence(ra.token_ids, rb.token_ids)
        sa = sb = None
        if check_state and ra.final_state is not None and rb.final_state is not None:
            sa, sb = ra.final_state, rb.final_state
        if div is not None or len(ra.token_ids) != len(rb.token_ids) or sa != sb:
            pos = div if div is not None else min(len(ra.token_ids),
                                                  len(rb.token_ids))
            out.append(Mismatch(
                label_a, label_b, rid, div,
                len(ra.token_ids), len(rb.token_ids), sa, sb,
                _margin_at(ra, pos), _margin_at(rb, pos),
                _classify(ra, rb, div, tau)))
    return out


def _verdict_of(mismatches: list[Mismatch]) -> str:
    if any(m.verdict == "fail" for m in mismatches):
        return "fail"
    if mismatches:
        return "boundary"
    return "pass"


def _forced_reject_opts(engine: Engine, opts: EngineOptions) -> EngineOptions:
    if CAP_DRAFT_MODE in _caps(engine):
        # exact: adversarial drafts + lossless standard verify -> all rejected
        return opts.with_(rejection_sample_method="standard", draft_mode="adversarial")
    # synthetic fallback: rates=[0]*K rejects every draft; emitted tokens are
    # still the target's greedy choices, so equivalence must hold regardless
    k = opts.num_speculative_tokens
    return opts.with_(rejection_sample_method="synthetic",
                      synthetic_acceptance_rates=(0.0,) * k)


def _forced_accept_opts(engine: Engine, opts: EngineOptions) -> EngineOptions:
    if CAP_DRAFT_MODE in _caps(engine):
        return opts.with_(rejection_sample_method="standard", draft_mode="oracle")
    # synthetic all-accept emits draft tokens, which are NOT target choices:
    # no output-equivalence oracle exists -> callers use natural acceptance
    return opts


# ---------------------------------------------------------------- checkers
# Single-repeat logic; run_case() aggregates over JudgeConfig.repeats.

def check_all_reject_equivalence(engine: Engine, case: CaseSpec,
                                 judge: JudgeConfig) -> InvariantOutcome:
    """I1. Every draft rejected <=> output equals target-only decoding.

    Bug class: rejection-path state handling (bonus token not folded into the
    recurrent state; SGLang #35150 forced-reject GDN drift)."""
    sc, opts = case.scenario, case.scenario.spec_opts
    o = InvariantOutcome("all_reject_equivalence", case.case_id,
                         sc.scenario_id, margin_tau=judge.tau)
    spec = run_arm(engine, sc, _forced_reject_opts(engine, opts),
                   label="spec_forced_reject")
    base = run_arm(engine, sc, _target_only(opts), label="target_only")
    o.arms = [_summary("spec_forced_reject", spec), _summary("target_only", base)]
    o.mismatches = _compare("spec_forced_reject", spec, "target_only", base,
                            CAP_STATE_DIGEST in _caps(engine), judge.tau)
    o.status = _verdict_of(o.mismatches)
    return o


def check_accept_path_equivalence(engine: Engine, case: CaseSpec,
                                  judge: JudgeConfig) -> InvariantOutcome:
    """I2. Under lossless (standard) verification the spec output equals
    target-only output for ANY acceptance pattern -- all-accept (oracle
    drafter) and natural mixed. Bug class: accept-path state fold errors."""
    sc, opts = case.scenario, case.scenario.spec_opts
    o = InvariantOutcome("accept_path_equivalence", case.case_id,
                         sc.scenario_id, margin_tau=judge.tau)
    fa = _forced_accept_opts(engine, opts)
    if fa is opts and CAP_DRAFT_MODE not in _caps(engine):
        fa = opts.with_(rejection_sample_method="standard")
    accept = run_arm(engine, sc, fa, label="spec_full_accept")
    natural = run_arm(engine, sc, opts.with_(rejection_sample_method="standard"),
                      label="spec_natural")
    base = run_arm(engine, sc, _target_only(opts), label="target_only")
    o.arms = [_summary("spec_full_accept", accept),
              _summary("spec_natural", natural), _summary("target_only", base)]
    o.mismatches = _compare("spec_full_accept", accept, "target_only", base,
                            CAP_STATE_DIGEST in _caps(engine), judge.tau)
    o.mismatches += _compare("spec_natural", natural, "target_only", base,
                             CAP_STATE_DIGEST in _caps(engine), judge.tau)
    o.status = _verdict_of(o.mismatches)
    return o


def check_permutation_invariance(engine: Engine, case: CaseSpec,
                                 judge: JudgeConfig) -> InvariantOutcome:
    """I3. Reordering the batch must not change any request's output."""
    sc, opts = case.scenario, case.scenario.spec_opts
    o = InvariantOutcome("permutation_invariance", case.case_id,
                         sc.scenario_id, margin_tau=judge.tau)
    n = len(sc.requests)
    orders = [tuple(range(n))]
    orders += [tuple(p) for p in sc.permutations if len(p) == n]
    if not sc.permutations:
        orders.append(tuple(reversed(range(n))))
    runs = {f"order{i}": run_arm(engine, sc, opts, order=order,
                                 label=f"order{i}")
            for i, order in enumerate(orders)}
    o.arms = [_summary(lbl, r) for lbl, r in runs.items()]
    labels = list(runs)
    for la, lb in zip(labels, labels[1:]):
        o.mismatches += _compare(la, runs[la], lb, runs[lb], False, judge.tau)
    o.status = _verdict_of(o.mismatches)
    return o


def check_split_merge_equivalence(engine: Engine, case: CaseSpec,
                                  judge: JudgeConfig) -> InvariantOutcome:
    """I4. Packed multi-request execution <=> independent per-request
    executions. Bug class: ragged/padded layout, compaction reorder."""
    sc, opts = case.scenario, case.scenario.spec_opts
    o = InvariantOutcome("split_merge_equivalence", case.case_id,
                         sc.scenario_id, margin_tau=judge.tau)
    packed = run_arm(engine, sc, opts, label="packed")
    singles = run_arm(engine, sc, opts, single=True, label="per_request")
    o.arms = [_summary("packed", packed), _summary("per_request", singles)]
    o.mismatches = _compare("packed", packed, "per_request", singles,
                            False, judge.tau)
    o.status = _verdict_of(o.mismatches)
    return o


def check_state_isolation(engine: Engine, case: CaseSpec,
                          judge: JudgeConfig) -> InvariantOutcome:
    """I5. Request A's acceptance/rejection must not influence request B:
    B alone == B beside A, in either slot order."""
    sc, opts = case.scenario, case.scenario.spec_opts
    o = InvariantOutcome("state_isolation", case.case_id,
                         sc.scenario_id, margin_tau=judge.tau)
    reqs = list(sc.requests)
    alone = {r.rid: run_arm(engine, sc, opts, order=(i,),
                            label=f"alone:{r.rid}")
             for i, r in enumerate(reqs)}
    o.arms = [_summary(f"alone:{r.rid}", alone[r.rid]) for r in reqs]
    for i, ra in enumerate(reqs):
        for j, rb in enumerate(reqs):
            if i >= j:
                continue
            for order, lbl in (((i, j), f"pair[{ra.rid}+{rb.rid}]"),
                               ((j, i), f"pair[{rb.rid}+{ra.rid}]")):
                pair = run_arm(engine, sc, opts, order=order, label=lbl)
                o.arms.append(_summary(lbl, pair))
                for r in (ra, rb):
                    o.mismatches += _compare(
                        lbl, {r.rid: pair[r.rid]}, f"alone:{r.rid}",
                        {r.rid: alone[r.rid][r.rid]}, False, judge.tau)
    o.status = _verdict_of(o.mismatches)
    return o


def check_graph_geometry_consistency(engine: Engine, case: CaseSpec,
                                     judge: JudgeConfig) -> InvariantOutcome:
    """I6. Graph regime changes must not change outputs (captured == eager
    == target-only). Bug class: capture/replay layout mismatch."""
    sc, opts = case.scenario, case.scenario.spec_opts
    o = InvariantOutcome("graph_geometry_consistency", case.case_id,
                         sc.scenario_id, margin_tau=judge.tau)
    captured = run_arm(engine, sc, opts.with_(graph_regime="captured"),
                       label="captured")
    eager = run_arm(engine, sc, opts.with_(graph_regime="eager"),
                    label="eager")
    base = run_arm(engine, sc, _target_only(opts), label="target_only")
    o.arms = [_summary("captured", captured), _summary("eager", eager),
              _summary("target_only", base)]
    o.mismatches = _compare("captured", captured, "eager", eager,
                            False, judge.tau)
    o.mismatches += _compare("captured", captured, "target_only", base,
                             False, judge.tau)
    o.status = _verdict_of(o.mismatches)
    return o


CHECKERS: dict[str, Callable[[Engine, CaseSpec, JudgeConfig],
                             InvariantOutcome]] = {
    "all_reject_equivalence": check_all_reject_equivalence,
    "accept_path_equivalence": check_accept_path_equivalence,
    "permutation_invariance": check_permutation_invariance,
    "split_merge_equivalence": check_split_merge_equivalence,
    "state_isolation": check_state_isolation,
    "graph_geometry_consistency": check_graph_geometry_consistency,
}


# ---------------------------------------------------------------- dispatch

def _aggregate(repeat_outcomes: Sequence[InvariantOutcome]) -> InvariantOutcome:
    """Majority vote across repeats, worst-wins ties (fail > boundary > pass);
    errors dominate only when they are the majority."""
    first = repeat_outcomes[0]
    verdicts = [o.status for o in repeat_outcomes]
    agg = InvariantOutcome(
        invariant=first.invariant, case_id=first.case_id,
        scenario_id=first.scenario_id,
        margin_tau=first.margin_tau, repeat_verdicts=verdicts)
    n = len(verdicts)
    errs = sum(1 for v in verdicts if v == "error")
    if errs * 2 > n:
        agg.status = "error"
        agg.details = first.details
        return agg
    fails = verdicts.count("fail")
    bounds = verdicts.count("boundary")
    if fails * 2 > n:
        agg.status = "fail"
    elif (fails + bounds) * 2 > n:
        agg.status = "boundary"
    else:
        agg.status = "pass"
    if errs:
        agg.details = f"{errs}/{n} repeats errored (excluded from vote)"
    # report mismatches from the worst non-error repeat (the deciding one)
    order = {"fail": 0, "boundary": 1, "pass": 2, "error": 3}
    deciding = min((o for o in repeat_outcomes if o.status != "error"),
                   key=lambda o: order.get(o.status, 9), default=first)
    agg.mismatches = deciding.mismatches
    agg.arms = deciding.arms
    if not agg.details:
        agg.details = deciding.details
    return agg


def run_case(engine: Engine, case: CaseSpec,
             judge: JudgeConfig | None = None) -> InvariantOutcome:
    """Dispatch one case; repeat per JudgeConfig.repeats and aggregate."""
    judge = judge or JudgeConfig()
    fn = CHECKERS.get(case.invariant)
    if fn is None:
        return InvariantOutcome(case.invariant, case.case_id,
                                case.scenario.scenario_id, "error",
                                details=f"unknown invariant '{case.invariant}'")
    outcomes = []
    for _ in range(judge.repeats):
        try:
            outcomes.append(fn(engine, case, judge))
        except Exception as e:  # noqa: BLE001 -- crash is a reportable outcome
            outcomes.append(InvariantOutcome(
                case.invariant, case.case_id, case.scenario.scenario_id,
                "error",
                details=f"{type(e).__name__}: {e}\n"
                        f"{traceback.format_exc(limit=4)}"))
    return outcomes[0] if len(outcomes) == 1 else _aggregate(outcomes)
