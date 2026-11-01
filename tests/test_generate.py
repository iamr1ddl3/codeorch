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


def test_sse_parsing_skips_noise_and_stops_at_done():
    lines = ["", ": keep-alive", 'data: {"a": 1}', "", 'data: {"a": 2}', "data: [DONE]", 'data: {"a": 3}']
    assert [e["a"] for e in g.iter_sse(lines, deadline=10**12)] == [1, 2]


def test_sse_deadline_raises_generation_timeout():
    with pytest.raises(g.GenerationTimeout):
        list(g.iter_sse(['data: {"a": 1}'], deadline=0))


def test_generation_timeout_is_recorded_and_charged_conservatively(tmp_path):
    def slow(*a):
        raise g.GenerationTimeout("still generating after 1200s")
    stats = g.run_batch("b", PROMPTS[:1], [SPEC], 1, 1000, 1, 1.0, "local", call=slow, runs_dir=tmp_path)
    rec = json.loads((tmp_path / "b" / "results.jsonl").read_text())
    assert rec["status"] == "gen_timeout" and rec["passed"] is False and rec["cost_estimated"] is True
    assert stats["spent_usd"] == pytest.approx(g.worst_case_cost(SPEC, g.SYSTEM_PROMPT, "Build a page."), abs=1e-4)


def test_budget_cut_drops_later_samples_not_later_prompts(tmp_path):
    worst = g.worst_case_cost(SPEC, g.SYSTEM_PROMPT, "Build a page.")
    call = fake_call(tin=10_000, tout=g.MAX_OUTPUT_TOKENS)  # each call costs >= worst case
    g.run_batch("b", PROMPTS, [SPEC], 2, budget_cap=worst * 3.5, workers=1,
                temperature=1.0, sandbox_mode="local", call=call, runs_dir=tmp_path)
    recs = [json.loads(l) for l in (tmp_path / "b" / "results.jsonl").read_text().splitlines()]
    assert sorted(r["prompt_id"] for r in recs) == ["P0", "P1", "P2"]  # every prompt got sample 0
    assert {r["sample"] for r in recs} == {0}


def test_parse_output_strict():
    files, fmt, inc = g.parse_output(GOOD)
    assert fmt == "strict" and list(files) == ["app/page.tsx"] and inc == []


def test_parse_output_unclosed_and_name_attribute():
    text = ('<file path="app/page.tsx">\nexport default function P() { return null; }\n'
            '<file name="lib/a.ts">\nexport const a = 1;\n')
    files, fmt, inc = g.parse_output(text, truncated=False)
    assert fmt == "unclosed" and set(files) == {"app/page.tsx", "lib/a.ts"} and inc == []
    assert files["lib/a.ts"] == "export const a = 1;\n"


def test_parse_output_unclosed_truncated_drops_last_block():
    text = '<file path="app/page.tsx">\nexport default 1;\n<file path="lib/a.ts">\nexport const a ='
    files, fmt, inc = g.parse_output(text, truncated=True)
    assert set(files) == {"app/page.tsx"} and inc == ["lib/a.ts"]


def test_parse_output_fenced_with_path_comment():
    text = ('Here you go.\n```css\n/* app/globals.css */\n@import "tailwindcss";\n```\n'
            '```tsx\n// app/page.tsx\nexport default function P() { return null; }\n```\n'
            '```bash\nnpm run dev\n```\n')
    files, fmt, inc = g.parse_output(text)
    assert fmt == "fenced" and set(files) == {"app/globals.css", "app/page.tsx"}
    assert files["app/globals.css"] == '@import "tailwindcss";\n'


def test_parse_output_prose_only_is_none():
    files, fmt, _ = g.parse_output("Here is a plan.\n```css\nbody { color: red; }\n```\n")
    assert files == {} and fmt == "none"


def test_unclosed_block_with_closed_fence_and_trailing_prose():
    text = ('<file path="app/page.tsx">\n```tsx\nconst s = `hi`;\nexport default function P() { return null; }\n```\n'
            'Now the next file.\n<file path="lib/a.ts">\nexport const a = 1;\n```\n')
    files, fmt, _ = g.parse_output(text)
    assert fmt == "unclosed"
    assert files["app/page.tsx"] == "const s = `hi`;\nexport default function P() { return null; }\n"
    assert files["lib/a.ts"] == "export const a = 1;\n"


def test_reparsed_records_do_not_double_count_spend(tmp_path):
    res = tmp_path / "results.jsonl"
    res.write_text('{"sample_id": "a", "status": "parse_error", "cost_usd": 1.0}\n'
                   '{"sample_id": "a", "status": "ok", "cost_usd": 1.0, "reparsed": true}\n'
                   '{"sample_id": "b", "status": "provider_error", "cost_usd": 0.0}\n')
    done, spent = g.load_done(res)
    assert done == {"a"} and spent == 1.0


def test_quota_exhaustion_aborts_the_batch(tmp_path):
    def quota(*a):
        raise RuntimeError('HTTP 429: {"error": {"message": "You exceeded your current quota, please check your plan"}}')
    stats = g.run_batch("b", PROMPTS, [SPEC], 1, 1000, 1, 1.0, "local", call=quota, runs_dir=tmp_path)
    assert stats["aborted"] and "quota" in stats["aborted"]


def test_glm_models_enabled_by_glm_key(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "DEEPSEEK_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("GLM_API_KEY", "x")
    specs = g.model_specs()
    assert set(specs) == {"glm_flash", "glm"}
    assert specs["glm_flash"].model_id == "glm-5.3-flash" and specs["glm_flash"].provider == "glm"
    assert g.PROVIDERS["glm"] is g.PROVIDERS["deepseek"]
    assert "coding" not in g.OPENAI_COMPAT["glm"][0]  # pay-as-you-go endpoint, not the Coding Plan


def test_zai_insufficient_balance_is_fatal():
    assert g._FATAL_RE.search('HTTP 429: {"error":{"code":"1113","message":"Insufficient balance or no resource package"}}')
