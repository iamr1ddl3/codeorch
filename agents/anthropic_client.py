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
