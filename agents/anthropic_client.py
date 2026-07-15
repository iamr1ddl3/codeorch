"""Thin Anthropic client wrapper.

Centralizes:
  - JSON-mode response extraction (we always want structured output)
  - Token usage capture so the Langfuse generation span gets cost data
  - Retry-once on transient errors (overloaded / rate limits)

Each agent calls call_model() inside its trace_agent(as_type='generation')
context — the returned usage dict is fed straight into span.update().
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any

from anthropic import Anthropic, APIStatusError, AsyncAnthropic


@dataclass
class ModelCall:
    text: str
    parsed: dict[str, Any] | None
    usage: dict[str, int]
    model: str
    latency_ms: int


_CLIENT: Anthropic | None = None
_ACLIENT: AsyncAnthropic | None = None


# ---------------------------------------------------------------------------
# Fake LLM backend (no API). Enabled with CODEORCH_FAKE_LLM=1.
#
# Lets the full pipeline run end-to-end against real Postgres + Langfuse with
# zero Anthropic spend — useful for wiring/store/observability smoke tests and
# offline demos. Canned per-stage JSON is dispatched off the system prompt
# (every agent's system starts "You are the <X> agent"). Returns the SAME
# ModelCall shape real calls return, so downstream parsing/validation is
# exercised unchanged. NOT a substitute for model-quality testing.
# ---------------------------------------------------------------------------

_FAKE_CODE = (
    "def is_even(n: int) -> bool:\n"
    "    return n % 2 == 0\n"
)
_FAKE_TESTS = (
    "from is_even import is_even\n\n"
    "def test_even():\n"
    "    assert is_even(4) is True\n\n"
    "def test_odd():\n"
    "    assert is_even(3) is False\n"
)
_FAKE_FINAL = f"# === is_even.py ===\n{_FAKE_CODE}"


def _fake_payload(system: str) -> dict[str, Any]:
    """Pick canned output by matching the agent's system prompt. The gate
    returns a passing score so the full review+doc path runs."""
    # Match on the "You are the <role>" anchor — agent bodies mention OTHER
    # agents' names (e.g. Reviewer's prompt references the Quality Gate), so a
    # plain substring scan misfires. Anchor to the role declaration.
    s = system.lower()
    if "you are the reviewer" in s:
        return {
            "approved": True,
            "notes": ["gate-approved; reviewer concurs"],
            "final_code": _FAKE_FINAL,
        }
    if "you are the documenter" in s:
        return {
            "documented_code": (
                '# === is_even.py ===\n'
                'def is_even(n: int) -> bool:\n'
                '    """Return True if n is even."""\n'
                '    return n % 2 == 0\n'
            ),
            "summary": "Returns whether an integer is even.",
        }
    if "you are the quality gate" in s:
        return {
            "score": 0.90,
            "verdict": "pass",
            "routing": "reviewer",
            "issues": [],
        }
    if "planner agent" in s:
        return {
            "tasks": [{
                "id": "T1",
                "description": "Implement is_even(n) returning n % 2 == 0",
                "language": "python",
                "complexity": "easy",
                "acceptance_criterion": "is_even(4) is True and is_even(3) is False",
            }],
            "acceptance_criteria": ["is_even returns True for even, False for odd"],
        }
    if "coder agent" in s:
        return {
            "files": {"is_even.py": _FAKE_CODE},
            "language": "python",
            "dependencies": [],
            "notes": "fake-llm canned implementation",
        }
    if "tester agent" in s:
        return {
            "test_files": {"test_is_even.py": _FAKE_TESTS},
            "framework": "pytest",
            "coverage_targets": ["is_even even input", "is_even odd input"],
        }
    if "orchestrator" in s:
        return {
            "summary": "Generated is_even with 2 passing tests; gate passed.",
            "verdict": "pass",
            "next_step_for_user": "Use the documented code as-is.",
        }
    # Unknown stage — return empty dict; the agent's own validator will raise,
    # which is the correct signal that a new stage needs a fake payload.
    return {}


def _fake_enabled() -> bool:
    return os.environ.get("CODEORCH_FAKE_LLM") == "1"


def _fake_call(model: str, system: str) -> "ModelCall":
    parsed = _fake_payload(system)
    text = json.dumps(parsed)
    return ModelCall(
        text=text,
        parsed=parsed,
        usage={"input": 0, "output": 0, "total": 0},
        model=f"fake:{model}",
        latency_ms=0,
    )


def _client() -> Anthropic:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _CLIENT


def _aclient() -> AsyncAnthropic:
    global _ACLIENT
    if _ACLIENT is None:
        _ACLIENT = AsyncAnthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _ACLIENT


def _parse(resp_content: Any, expect_json: bool) -> tuple[str, dict[str, Any] | None]:
    text = "".join(
        block.text for block in resp_content if getattr(block, "type", None) == "text"
    )
    if not expect_json:
        return text, None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        if cleaned.startswith("json\n"):
            cleaned = cleaned[5:]
    try:
        return text, json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"model returned non-JSON: {exc}: {text[:200]!r}") from exc


def call_model(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    expect_json: bool = True,
) -> ModelCall:
    """One Anthropic call. Returns text + parsed JSON + token usage.

    expect_json=True parses the response body as JSON and raises ValueError
    if the model returned malformed output (caught by base_agent and turned
    into a partial-state record).

    Note: temperature is intentionally not passed — Opus 4.7 rejects it
    (deprecated for that model). Sonnet/Haiku still accept it but the default
    server-side temperature is what we want here.
    """
    if _fake_enabled():
        return _fake_call(model, system)
    started_at = time.monotonic()
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            resp = _client().messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            break
        except APIStatusError as exc:
            # 429 / 529 are worth one retry; others fail fast.
            if exc.status_code in (429, 529) and attempt == 1:
                last_exc = exc
                time.sleep(1.0)
                continue
            raise
    else:
        raise last_exc  # type: ignore[misc]

    text, parsed = _parse(resp.content, expect_json)

    return ModelCall(
        text=text,
        parsed=parsed,
        usage={
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
            "total": resp.usage.input_tokens + resp.usage.output_tokens,
        },
        model=model,
        latency_ms=int((time.monotonic() - started_at) * 1000),
    )


async def acall_model(
    *,
    model: str,
    system: str,
    user: str,
    max_tokens: int = 2048,
    expect_json: bool = True,
) -> ModelCall:
    """Async twin of call_model(). Used inside asyncio.gather() for the
    Coder + Tester parallel fan-out (Day 5)."""
    import asyncio

    if _fake_enabled():
        return _fake_call(model, system)
    started_at = time.monotonic()
    last_exc: Exception | None = None
    for attempt in (1, 2):
        try:
            resp = await _aclient().messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            break
        except APIStatusError as exc:
            if exc.status_code in (429, 529) and attempt == 1:
                last_exc = exc
                await asyncio.sleep(1.0)
                continue
            raise
    else:
        raise last_exc  # type: ignore[misc]

    text, parsed = _parse(resp.content, expect_json)
    return ModelCall(
        text=text,
        parsed=parsed,
        usage={
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
            "total": resp.usage.input_tokens + resp.usage.output_tokens,
        },
        model=model,
        latency_ms=int((time.monotonic() - started_at) * 1000),
    )
