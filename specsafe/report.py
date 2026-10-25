"""Run records and report rendering (JSONL + markdown matrix), v2:
three-tier verdicts PASS / BOUNDARY / FAIL (+ error)."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .invariants import InvariantOutcome

_STATUS_ORDER = {"pass": 0, "boundary": 1, "fail": 2, "error": 3}


@dataclass
class RunRecord:
    run_id: str
    matrix: str
    engine: str
    engine_info: dict
    ts: str
    duration_s: float
    outcomes: list[InvariantOutcome] = field(default_factory=list)
    margin_tau: float | None = None
    repeats: int | None = None

    def _count(self, status: str) -> int:
        return sum(1 for o in self.outcomes if o.status == status)

    @property
    def n_pass(self) -> int:
        return self._count("pass")

    @property
    def n_boundary(self) -> int:
        return self._count("boundary")

    @property
    def n_fail(self) -> int:
        return self._count("fail")

    @property
    def n_error(self) -> int:
        return self._count("error")

    @property
    def ok(self) -> bool:
        return self.n_fail == 0 and self.n_error == 0


def write_report(out_dir: Path, record: RunRecord) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    md_path = out_dir / "summary.md"

    json_path.write_text(json.dumps(asdict(record), indent=2, ensure_ascii=False),
                         encoding="utf-8")

    judge = ""
    if record.margin_tau is not None:
        judge = (f", margin tau={record.margin_tau}"
                 + (f", {record.repeats}x repeats" if record.repeats else ""))
    lines = [
        f"# SpecSafe run `{record.run_id}`",
        "",
        f"- matrix: **{record.matrix}**",
        f"- engine: **{record.engine}**{judge}",
        f"- started: {record.ts}  duration: {record.duration_s:.1f}s",
        f"- verdict: **{'ALL PASS' if record.ok else 'FAILURES PRESENT'}** "
        f"({record.n_pass} pass / {record.n_boundary} boundary / "
        f"{record.n_fail} fail / {record.n_error} error)",
        "",
        "Verdict legend: PASS = token-identical; BOUNDARY = divergence only "
        "at near-tie positions (margin <= tau, numeric floor); "
        "FAIL = confident divergence (margin > tau both sides).",
        "",
        "## Engine",
        "",
    ]
    for k, v in record.engine_info.items():
        lines.append(f"- {k}: `{v}`")
    lines += ["", "## Results", "",
              "| case | invariant | status | detail |", "|---|---|---|---|"]
    for o in sorted(record.outcomes, key=lambda x: _STATUS_ORDER.get(x.status, 9)):
        detail = o.details if o.status == "error" else (
            "; ".join(m.render() for m in o.mismatches[:3])
            + (" …" if len(o.mismatches) > 3 else ""))
        if o.repeat_verdicts and len(o.repeat_verdicts) > 1:
            detail = f"[repeats {','.join(o.repeat_verdicts)}] " + detail
        detail = detail.replace("|", "\\|").replace("\n", " ")[:400]
        lines.append(f"| {o.case_id} | {o.invariant} | **{o.status}** | {detail} |")
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return json_path, md_path


def new_run_id(matrix: str) -> str:
    return f"{time.strftime('%Y%m%d_%H%M%S')}-{matrix}"
