"""Scenario serialization, matrix determinism, report writing, CLI smoke."""
import json

from specsafe.cli import main
from specsafe.generator import build_matrix
from specsafe.report import RunRecord, write_report
from specsafe.scenario import CaseSpec, Scenario


def test_scenario_json_roundtrip(tmp_path):
    case = build_matrix("smoke")[0]
    p = tmp_path / "s.json"
    p.write_text(case.scenario.to_json(), encoding="utf-8")
    sc2 = Scenario.from_json(p.read_text(encoding="utf-8"))
    assert sc2.to_dict() == case.scenario.to_dict()
    c2 = CaseSpec.from_dict(case.to_dict())
    assert c2.to_dict() == case.to_dict()


def test_matrix_generation_is_deterministic():
    assert ([c.to_dict() for c in build_matrix("smoke")] ==
            [c.to_dict() for c in build_matrix("smoke")])
    assert ([c.to_dict() for c in build_matrix("static32")] ==
            [c.to_dict() for c in build_matrix("static32")])


def test_matrix_catalog():
    for name in ("selftest", "smoke", "static32", "drift35150"):
        cases = build_matrix(name)
        assert cases, name
        for c in cases:
            assert c.invariant in {
                "all_reject_equivalence", "accept_path_equivalence",
                "permutation_invariance", "split_merge_equivalence",
                "state_isolation", "graph_geometry_consistency"}
    assert len(build_matrix("static32")) >= 30
    assert len(build_matrix("drift35150")) == 3


def test_cli_gen_and_run(tmp_path, capsys):
    rc = main(["gen", "--matrix", "smoke", "--out", str(tmp_path / "m.json")])
    assert rc == 0
    payload = json.loads((tmp_path / "m.json").read_text(encoding="utf-8"))
    assert payload["matrix"] == "smoke" and payload["n_cases"] > 0

    out_dir = tmp_path / "run"
    rc = main(["run", "--engine", "reference", "--matrix", "selftest",
               "--out", str(out_dir)])
    assert rc == 0
    rep = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert rep["matrix"] == "selftest"
    assert all(o["status"] == "pass" for o in rep["outcomes"])
    md = (out_dir / "summary.md").read_text(encoding="utf-8")
    assert "ALL PASS" in md and "| selftest-" in md


def test_cli_run_failure_exit_code(tmp_path):
    rc = main(["run", "--engine", "mutant:RejectBonusDrop", "--matrix",
               "selftest", "--out", str(tmp_path / "bad")])
    assert rc == 1


def test_report_record_writes(tmp_path):
    from specsafe.engine import ReferenceEngine
    from specsafe.invariants import run_case
    outcomes = [run_case(ReferenceEngine(), c) for c in build_matrix("selftest")]
    rec = RunRecord(run_id="x", matrix="selftest", engine="reference",
                    engine_info={"a": 1}, ts="t", duration_s=1.0,
                    outcomes=outcomes)
    jp, mp = write_report(tmp_path, rec)
    assert jp.exists() and mp.exists()
    data = json.loads(jp.read_text(encoding="utf-8"))
    assert len(data["outcomes"]) == 6
