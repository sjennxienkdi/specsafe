"""SpecSafe command-line interface (oracle v2: margin-aware, repeat-majority).

Local (no GPU, no deps):
    specsafe selftest                    # mutants FAIL + noise double BOUNDARY + clean PASS
    specsafe run --engine reference --matrix smoke --out results/smoke-ref
    specsafe gen --matrix static32 --out scenarios/static32.json

Cloud box (vLLM installed, GPU free):
    specsafe run --engine vllm --matrix floor --model <path> \
        --margin-tau 0.5 --repeats 1 --out results/floor
    specsafe run --engine vllm --matrix static32 --margin-tau 0.5 --repeats 3 \
        --model <path> --out results/static32-v2
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

from . import __version__, floor  # noqa: F401  (floor registers its checkers)
from .engine import make_engine
from .engine.mutants import MUTANTS, NoisyFlip
from .generator import MATRICES, build_matrix
from .invariants import (
    JudgeConfig,
    _compare,
    _verdict_of,
    run_arm,
    run_case,
)
from .report import RunRecord, new_run_id, write_report
from .scenario import EngineOptions, ReqSpec, Scenario


def _cmd_selftest(args) -> int:
    """For each invariant: clean engine passes, matching mutant FAILs under
    the margin-aware oracle; plus the oracle-honesty double: NoisyFlip
    (near-tie flips only) MUST be judged BOUNDARY, not FAIL."""
    from .engine import ReferenceEngine

    cases = build_matrix("selftest")
    bad = 0
    print(f"specsafe {__version__} selftest: margin-aware oracle v2\n")
    print("== 1. mutant detection (real bugs must FAIL, tau=0.5) ==")
    for inv, mutant_cls in MUTANTS.items():
        case = next(c for c in cases if c.invariant == inv)
        clean = run_case(ReferenceEngine(), case, JudgeConfig(tau=0.5))
        mut = run_case(mutant_cls(), case, JudgeConfig(tau=0.5))
        ok_clean = clean.status == "pass"
        ok_mut = mut.status == "fail"
        mark = "OK " if (ok_clean and ok_mut) else "!! "
        if not (ok_clean and ok_mut):
            bad += 1
        print(f"{mark}{inv:28s} clean={clean.status:7s} "
              f"{mutant_cls.name}={mut.status}")
        if mut.status != "fail":
            print(f"    !! mutant escaped detection: {mutant_cls.name}")
        if clean.status != "pass":
            for m in clean.mismatches[:3]:
                print(f"    clean-engine mismatch: {m.render()}")

    print("\n== 2. oracle honesty (noise double must be BOUNDARY, not FAIL) ==")
    noise = _noise_double_check()
    if noise["tau05"] == "boundary" and noise["tau0"] == "fail" \
            and noise["n_divergent"] > 0:
        print(f"OK  NoisyFlip: {noise['n_divergent']}/{noise['n_requests']} "
              f"requests diverge at near-tie positions -> "
              f"tau=0.5:{noise['tau05']} / tau=0:{noise['tau0']}")
    else:
        bad += 1
        print(f"!!  NoisyFlip check unexpected: {noise}")

    print(f"\n{'SELFTEST PASS' if bad == 0 else f'SELFTEST FAILED ({bad})'}")
    return 0 if bad == 0 else 1


def _noise_double_check() -> dict:
    """Clean vs NoisyFlip on a long scenario: every flip happens at a
    near-tie position by construction, so a correct oracle calls it
    BOUNDARY at tau=0.5 and FAIL at tau=0."""
    from .engine import ReferenceEngine

    sc = Scenario(
        "selftest/noise-double", "oracle honesty double",
        tuple(ReqSpec(f"r{i:03d}", f"noise-double-prompt-{i}", 128)
              for i in range(3)),
        EngineOptions(method="none", seed=99), (), 99)
    a = run_arm(ReferenceEngine(), sc, sc.spec_opts, label="clean")
    b = run_arm(NoisyFlip(), sc, sc.spec_opts, label="noisy")
    m05 = _compare("clean", a, "noisy", b, False, 0.5)
    m00 = _compare("clean", a, "noisy", b, False, 0.0)
    return {
        "n_requests": len(a),
        "n_divergent": len(m05),
        "tau05": _verdict_of(m05),
        "tau0": _verdict_of(m00),
        "first_margin": m05[0].margin_a if m05 else None,
    }


def _cmd_run(args) -> int:
    matrix = build_matrix(args.matrix)
    if args.filter:
        matrix = [c for c in matrix if args.filter in c.case_id]
        if not matrix:
            print(f"no cases match --filter {args.filter!r}", file=sys.stderr)
            return 2
    judge = JudgeConfig(tau=args.margin_tau, repeats=args.repeats)
    engine_kwargs = {}
    if args.engine == "vllm":
        engine_kwargs.update(
            model=args.model, max_model_len=args.max_model_len,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_mem,
            max_num_seqs=args.max_num_seqs,
            default_spec_method=args.spec_method)
        if args.kv:
            engine_kwargs["extra_llm_kwargs"] = dict(
                _parse_kv(p) for p in args.kv)
    engine = make_engine(args.engine, **engine_kwargs)
    run_id = new_run_id(args.matrix)
    t0 = time.time()
    ts = _dt.datetime.now().isoformat(timespec="seconds")
    outcomes = []
    for i, case in enumerate(matrix, 1):
        t1 = time.time()
        out = run_case(engine, case, judge)
        outcomes.append(out)
        rep = (f"[{','.join(out.repeat_verdicts)}]" if out.repeat_verdicts else "")
        print(f"[{i:3d}/{len(matrix)}] {case.case_id:36s} "
              f"{out.invariant:28s} {out.status:7s}{rep} {time.time() - t1:6.1f}s")
        if out.status not in ("pass", "boundary"):
            for m in out.mismatches[:3]:
                print(f"         {m.render()}")
        if out.status == "boundary":
            n_b = sum(1 for m in out.mismatches if m.verdict == "boundary")
            print(f"         ({n_b} near-tie divergence(s) within floor)")
    if args.engine == "vllm":
        try:
            engine.shutdown()
        except Exception:  # noqa: BLE001
            pass
    rec = RunRecord(run_id=run_id, matrix=args.matrix,
                    engine=engine.info().name, engine_info=engine.info().details,
                    ts=ts, duration_s=round(time.time() - t0, 1),
                    outcomes=outcomes, margin_tau=judge.tau,
                    repeats=judge.repeats)
    jp, mp = write_report(Path(args.out), rec)
    print(f"\nverdict: {'ALL PASS' if rec.ok else 'FAILURES PRESENT'} "
          f"({rec.n_pass}/{len(outcomes)} pass, {rec.n_boundary} boundary, "
          f"{rec.n_fail} fail, {rec.n_error} error)")
    print(f"report: {jp}\n        {mp}")
    return 0 if rec.ok else 1


def _cmd_gen(args) -> int:
    cases = build_matrix(args.matrix)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "matrix": args.matrix,
        "n_cases": len(cases),
        "cases": [c.to_dict() for c in cases],
    }
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"wrote {len(cases)} cases -> {out}")
    return 0


def _parse_kv(s: str) -> tuple[str, object]:
    k, _, v = s.partition("=")
    low = v.lower()
    if low in ("true", "false"):
        return k, low == "true"
    for cast in (int, float):
        try:
            return k, cast(v)
        except ValueError:
            pass
    return k, v


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="specsafe",
                                description="speculative-decoding correctness harness")
    p.add_argument("--version", action="version", version=f"specsafe {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("selftest", help="prove oracle v2 catches bugs and spares noise (local)")
    sp.set_defaults(fn=_cmd_selftest)

    rp = sub.add_parser("run", help="run an invariant matrix on an engine")
    rp.add_argument("--engine", required=True,
                    help="reference | vllm | mutant:<ClassName> | double:<ClassName>")
    rp.add_argument("--matrix", required=True, choices=sorted(MATRICES))
    rp.add_argument("--out", required=True, help="output dir for report.json/summary.md")
    rp.add_argument("--filter", default="", help="substring filter on case_id")
    rp.add_argument("--margin-tau", type=float, default=0.5,
                    help="top1-top2 gap (nats) below which a divergence is "
                         "numeric-boundary, not failure (default 0.5)")
    rp.add_argument("--repeats", type=int, default=1,
                    help="repeat each case N times, majority vote "
                         "(3 recommended on GPU)")
    rp.add_argument("--model", default="Qwen/Qwen3.5-0.8B",
                    help="[vllm] model path or repo id")
    rp.add_argument("--spec-method", default="mtp",
                    help="[vllm] method used for the engine-agnostic 'spec' "
                         "alias (mtp | dspark | ngram | ...)")
    rp.add_argument("--max-model-len", type=int, default=2048)
    rp.add_argument("--max-num-seqs", type=int, default=64,
                    help="[vllm] GDN hybrids need 1 Mamba block per decode "
                         "seq; keep below available blocks (default 64)")
    rp.add_argument("--tp", type=int, default=1)
    rp.add_argument("--gpu-mem", type=float, default=0.85)
    rp.add_argument("--kv", action="append", default=[],
                    help="[vllm] extra LLM kwargs, e.g. --kv dtype=float16 "
                         "(repeatable)")
    rp.set_defaults(fn=_cmd_run)

    gp = sub.add_parser("gen", help="emit a matrix as scenario JSON")
    gp.add_argument("--matrix", required=True, choices=sorted(MATRICES))
    gp.add_argument("--out", required=True)
    gp.set_defaults(fn=_cmd_gen)

    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
