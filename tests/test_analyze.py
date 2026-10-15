"""Tests for benchmark/analyze.py statistics and categorisation."""

from __future__ import annotations

import pytest

from benchmark import analyze as a


def test_pass_at_k_matches_known_values():
    assert a.pass_at_k(3, 0, 1) == 0.0
    assert a.pass_at_k(3, 3, 1) == 1.0
    assert a.pass_at_k(3, 1, 1) == pytest.approx(1 / 3)
    assert a.pass_at_k(3, 1, 3) == 1.0          # one pass among 3 draws of 3
    assert a.pass_at_k(4, 1, 2) == pytest.approx(0.5)


def test_mcnemar_exact():
    assert a.mcnemar_exact(0, 0) == 1.0
    assert a.mcnemar_exact(5, 5) == 1.0
    assert a.mcnemar_exact(0, 6) == pytest.approx(2 / 64)   # 2 * 0.5**6


def test_categories_follow_causal_order():
    assert a.category({"passed": True}) == "pass"
    assert a.category({"status": "parse_error"}) == "parse_error"
    assert a.category({"status": "ok", "truncated": True, "step_passed": {"tsc": False}}) == "truncated"
    assert a.category({"status": "ok", "step_passed": {"tsc": False, "eslint": False, "build": False}}) == "type_error"
    assert a.category({"status": "ok", "step_passed": {"tsc": True, "eslint": True, "build": False}}) == "build_error"
    assert a.category({"status": "ok", "step_passed": {"tsc": True, "eslint": False, "build": True}}) == "lint_only"


def test_bootstrap_ci_brackets_point_estimate():
    by_prompt = {f"p{i}": [i % 2 == 0] * 3 for i in range(20)}
    lo, hi = a.bootstrap_ci(by_prompt, a.mean_pass)
    assert lo <= 0.5 <= hi and 0 <= lo and hi <= 1


def test_load_prefers_real_result_over_provider_error(tmp_path, monkeypatch):
    runs = tmp_path / "runs" / "b"
    runs.mkdir(parents=True)
    (runs / "results.jsonl").write_text(
        '{"sample_id": "x", "status": "provider_error"}\n{"sample_id": "x", "status": "ok", "passed": true}\n'
        '{"sample_id": "y", "status": "ok", "passed": false}\n{"sample_id": "y", "status": "provider_error"}\n')
    monkeypatch.setattr(a, "HERE", tmp_path)
    got = {r["sample_id"]: r["status"] for r in a.load("b")}
    assert got == {"x": "ok", "y": "ok"}
