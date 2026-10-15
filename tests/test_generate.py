"""Offline tests for benchmark/generate.py (fake provider, fake sandbox)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from benchmark import generate as g

SPEC = g.ModelSpec("fake", "fake", "fake-1", price_in=1.0, price_out=5.0)
PROMPTS = [{"id": f"P{i}", "prompt": "Build a page.", "tier": "low", "source": "test", "category": "x"}
           for i in range(3)]
GOOD = '<file path="app/page.tsx">\nexport default function P() { return <main/>; }\n</file>'


def fake_call(text=GOOD, stop="end_turn", tin=1000, tout=2000):
    calls = []

    def call(spec, system, user, temperature):
        calls.append(user)
        return g.Generation(text, tin, tout, stop, 5, spec.model_id)
    call.calls = calls
    return call


@pytest.fixture(autouse=True)
def fake_sandbox(monkeypatch):
    res = SimpleNamespace(status="passed", passed=True, total_errors=0, steps=[],
                          rejected_files=[], wall_ms=1, run_id="r")
    monkeypatch.setattr(g, "run_build_check", lambda files, mode=None, run_id=None: res)


def test_parse_file_blocks_and_tolerates_fences():
    text = ('intro\n<file path="app/page.tsx">\n```tsx\nexport default 1\n```\n</file>\n'
            '<file path="lib/a.ts">\nexport const a = 1;\n</file>')
    assert g.parse_files(text) == {"app/page.tsx": "export default 1\n", "lib/a.ts": "export const a = 1;\n"}


def test_parse_returns_empty_for_prose():
    assert g.parse_files("Sure! Here is how you could build it...") == {}


def test_budget_stops_when_actual_spend_leaves_no_room(tmp_path):
    worst = g.worst_case_cost(SPEC, g.SYSTEM_PROMPT, "Build a page.")
    call = fake_call(tin=10_000, tout=g.MAX_OUTPUT_TOKENS)  # actual cost >= worst case
    stats = g.run_batch("b", PROMPTS, [SPEC], 1, budget_cap=worst * 1.5, workers=1,
                        temperature=1.0, sandbox_mode="local", call=call, runs_dir=tmp_path)
    assert len(call.calls) == 1 and stats["skipped_budget"] == 2
    assert stats["spent_usd"] <= worst * 1.5


def test_budget_holds_under_parallel_workers(tmp_path):
    import time as _t
    worst = g.worst_case_cost(SPEC, g.SYSTEM_PROMPT, "Build a page.")
    inner = fake_call(tin=10_000, tout=g.MAX_OUTPUT_TOKENS)

    def slow(*a):
        _t.sleep(0.3)  # keep the first call in flight while others try to reserve
        return inner(*a)
    stats = g.run_batch("b", PROMPTS, [SPEC], 1, budget_cap=worst * 1.5, workers=3,
                        temperature=1.0, sandbox_mode="local", call=slow, runs_dir=tmp_path)
    assert len(inner.calls) == 1 and stats["skipped_budget"] == 2


def test_cost_accounting_and_resume(tmp_path):
    call = fake_call(tin=1_000_000, tout=100_000)  # $1.00 in + $0.50 out
    g.run_batch("b", PROMPTS[:2], [SPEC], 1, 1000, 1, 1.0, "local", call=call, runs_dir=tmp_path)
    recs = [json.loads(l) for l in (tmp_path / "b" / "results.jsonl").read_text().splitlines()]
    assert [r["cost_usd"] for r in recs] == [1.5, 1.5]

    call2 = fake_call()
    stats = g.run_batch("b", PROMPTS, [SPEC], 1, 1000, 1, 1.0, "local", call=call2, runs_dir=tmp_path)
    assert len(call2.calls) == 1 and stats["already_done"] == 2  # only P2 is new
    assert stats["spent_usd"] >= 3.0  # prior spend still counts toward the cap


def test_unparseable_response_is_a_recorded_failure(tmp_path):
    g.run_batch("b", PROMPTS[:1], [SPEC], 1, 1000, 1, 1.0, "local",
                call=fake_call(text="I cannot do that."), runs_dir=tmp_path)
    rec = json.loads((tmp_path / "b" / "results.jsonl").read_text())
    assert rec["status"] == "parse_error" and rec["build_passed"] is False


def test_truncation_is_flagged(tmp_path):
    g.run_batch("b", PROMPTS[:1], [SPEC], 1, 1000, 1, 1.0, "local",
                call=fake_call(stop="max_tokens"), runs_dir=tmp_path)
    assert json.loads((tmp_path / "b" / "results.jsonl").read_text())["truncated"] is True


def test_provider_error_is_recorded_free_and_retried_on_resume(tmp_path):
    def boom(*a):
        raise RuntimeError("overloaded")
    g.run_batch("b", PROMPTS[:1], [SPEC], 1, 1000, 1, 1.0, "local", call=boom, runs_dir=tmp_path)
    rec = json.loads((tmp_path / "b" / "results.jsonl").read_text())
    assert rec["status"] == "provider_error" and rec["cost_usd"] == 0.0
    call = fake_call()
    g.run_batch("b", PROMPTS[:1], [SPEC], 1, 1000, 1, 1.0, "local", call=call, runs_dir=tmp_path)
    assert len(call.calls) == 1


def test_models_enabled_only_by_their_keys(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY", "GEMINI_MODEL", "GEMINI_PRICE_IN"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("GEMINI_PRICE_IN", "2.5")
    specs = g.model_specs()
    assert set(specs) == {"gemini"}
    assert specs["gemini"].price_in == 2.5 and specs["gemini"].max_out == g.THINKING_MAX_OUT


def test_length_finish_reason_counts_as_truncation(tmp_path):
    g.run_batch("b", PROMPTS[:1], [SPEC], 1, 1000, 1, 1.0, "local",
                call=fake_call(stop="length"), runs_dir=tmp_path)
    assert json.loads((tmp_path / "b" / "results.jsonl").read_text())["truncated"] is True


def test_billing_error_aborts_the_batch(tmp_path):
    calls = []

    def broke(*a):
        calls.append(1)
        raise RuntimeError("Error code: 400 - Your credit balance is too low to access the Anthropic API.")
    stats = g.run_batch("b", PROMPTS, [SPEC], 2, 1000, 1, 1.0, "local", call=broke, runs_dir=tmp_path)
    assert len(calls) == 1 and stats["aborted"] and "credit balance" in stats["aborted"]


def test_truncated_output_that_still_builds_is_not_a_pass(tmp_path):
    # The cut-off file is dropped by the parser, the rest builds: must not count as passed.
    text = GOOD + '\n<file path="components/InvoiceForm.tsx">\nexport function InvoiceForm() {\n  return ('
    g.run_batch("b", PROMPTS[:1], [SPEC], 1, 1000, 1, 1.0, "local",
                call=fake_call(text=text, stop="length"), runs_dir=tmp_path)
    rec = json.loads((tmp_path / "b" / "results.jsonl").read_text())
    assert rec["build_passed"] is True and rec["passed"] is False
    assert rec["incomplete_files"] == ["components/InvoiceForm.tsx"]


def test_complete_output_that_builds_passes(tmp_path):
    g.run_batch("b", PROMPTS[:1], [SPEC], 1, 1000, 1, 1.0, "local", call=fake_call(), runs_dir=tmp_path)
    rec = json.loads((tmp_path / "b" / "results.jsonl").read_text())
    assert rec["passed"] is True and rec["incomplete_files"] == []
