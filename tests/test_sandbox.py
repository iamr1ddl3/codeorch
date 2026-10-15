"""Tests for sandbox/runner.py.

Unit tests (path validation, parsing, honest-failure) run anywhere.
Integration tests build the fixtures for real; they need Node and
`npm ci` in sandbox/template and are skipped otherwise (~30 s total).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from sandbox import runner

FIXTURES = Path(__file__).parent / "fixtures" / "sandbox"
HAS_TOOLCHAIN = shutil.which("node") is not None and (runner.TEMPLATE_DIR / "node_modules").exists()
needs_toolchain = pytest.mark.skipif(not HAS_TOOLCHAIN, reason="node + sandbox/template/node_modules required")


def _load(name: str) -> dict[str, str]:
    return runner._load_input(FIXTURES / name)


# ---------------- unit ----------------

@pytest.mark.parametrize("path,reason", [
    ("/etc/passwd", "absolute path"),
    ("../escape.ts", "path traversal"),
    ("app/../../x.ts", "path traversal"),
    ("node_modules/next/index.js", "writes into node_modules/"),
    ("package.json", "protected toolchain file"),
    ("./tsconfig.json", "protected toolchain file"),
])
def test_rejects_unsafe_or_protected_paths(path, reason):
    ok, rejected = runner.validate_files({path: "x"})
    assert ok == {}
    assert rejected == [{"path": path, "reason": reason}]


def test_normalises_leading_dot_slash():
    ok, rejected = runner.validate_files({"./app/page.tsx": "export default 1"})
    assert ok == {"app/page.tsx": "export default 1"} and rejected == []


def test_empty_output_is_a_failure_not_a_pass():
    res = runner.run_build_check({"app/page.tsx": "   "}, mode="local", run_id="test-empty")
    assert res.status == "empty_output" and not res.passed and res.steps == []


def test_parse_tsc_lines():
    n, errs = runner._parse_tsc("app/page.tsx(11,9): error TS2322: Type 'string' is not assignable to type 'number'.\n")
    assert n == 1 and errs[0]["code"] == "TS2322" and errs[0]["line"] == 11


def test_parse_build_counts_root_causes_once():
    log = ("Failed to compile.\n\n./app/page.tsx\nModule not found: Can't resolve 'recharts'\n\n"
           "> Build failed because of webpack errors\n")
    n, errs = runner._parse_build(log, exit_code=1)
    assert n == 1 and "recharts" in errs[0]["message"]


# ---------------- integration ----------------

@needs_toolchain
def test_good_fixture_passes_all_three_steps():
    res = runner.run_build_check(_load("good_pricing"), mode="local", run_id="test-good")
    assert res.status == "passed", json.dumps(res.to_dict(), indent=2)[:2000]
    assert [s.step for s in res.steps] == ["tsc", "eslint", "build"]
    assert all(s.passed for s in res.steps)


@needs_toolchain
def test_broken_fixture_is_caught_by_every_step():
    res = runner.run_build_check(_load("broken_app"), mode="local", run_id="test-broken")
    by = {s.step: s for s in res.steps}
    assert res.status == "failed"
    assert {e["code"] for e in by["tsc"].errors} >= {"TS2307", "TS2322"}  # missing module, type error
    assert any(e["code"] == "react-hooks/rules-of-hooks" for e in by["eslint"].errors)
    assert by["build"].error_count == 1  # one root cause: missing 'recharts'
    assert res.rejected_files == [{"path": "package.json", "reason": "protected toolchain file"}]
    assert (Path(res.run_dir) / "result.json").exists()
