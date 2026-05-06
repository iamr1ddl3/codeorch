"""Promptfoo benchmark runner scaffold.

Day 3 ships the runner shell and HTML report generator only.
Day 6 wires the actual 10-task golden benchmark (3 easy / 4 medium / 3 hard).
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

REPORT_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>CodeOrch Eval Report — {timestamp}</title>
<style>
  body {{ font-family: ui-monospace, monospace; max-width: 900px; margin: 2em auto; padding: 0 1em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
  .pass {{ color: #1a7f1a; }}
  .fail {{ color: #b00020; }}
</style>
</head>
<body>
<h1>CodeOrch Eval Report</h1>
<p><strong>Run:</strong> {timestamp}</p>
<p><strong>Pass rate:</strong> {pass_rate:.1%} ({passes} / {total})</p>
<table>
<tr><th>Task</th><th>Difficulty</th><th>Score</th><th>Result</th></tr>
{rows}
</table>
</body>
</html>
"""


@dataclass
class TaskResult:
    task_id: str
    difficulty: str
    score: float
    passed: bool


def run_promptfoo(config_path: Path) -> list[TaskResult]:
    """Invoke promptfoo CLI and parse JSON output.

    Day 3: stub. Day 6 wires real config + assertions.
    """
    if not config_path.exists():
        return []
    proc = subprocess.run(
        ["promptfoo", "eval", "--config", str(config_path), "--output", "json"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        return []
    payload = json.loads(proc.stdout)
    return [
        TaskResult(
            task_id=row["test"]["description"],
            difficulty=row["test"].get("metadata", {}).get("difficulty", "unknown"),
            score=row.get("score", 0.0),
            passed=row.get("success", False),
        )
        for row in payload.get("results", [])
    ]


def render_html(results: list[TaskResult], output_path: Path) -> None:
    rows = "\n".join(
        f'<tr><td>{r.task_id}</td><td>{r.difficulty}</td>'
        f'<td>{r.score:.2f}</td>'
        f'<td class="{"pass" if r.passed else "fail"}">{"PASS" if r.passed else "FAIL"}</td></tr>'
        for r in results
    )
    passes = sum(1 for r in results if r.passed)
    total = len(results) or 1
    output_path.write_text(
        REPORT_TEMPLATE.format(
            timestamp=datetime.utcnow().isoformat(timespec="seconds"),
            pass_rate=passes / total,
            passes=passes,
            total=total,
            rows=rows or "<tr><td colspan='4'><em>No results yet — wire benchmark in Day 6.</em></td></tr>",
        )
    )


if __name__ == "__main__":
    config = Path("evals/promptfoo.yaml")
    out = Path("evals/report.html")
    render_html(run_promptfoo(config), out)
    print(f"wrote {out}")
