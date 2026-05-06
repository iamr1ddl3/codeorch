"""Langfuse v3+ instrumentation for CodeOrch.

Trace structure (per Langfuse best practices):
    one run = one trace = one outer span
    one agent call = one child span (or generation, when wrapping an LLM call)

Trace attributes (set once at run start, propagated to every child span):
    - session_id  = str(run_id) — groups all agents of one run in Sessions view
    - user_id     = caller-supplied (defaults to "anon" until auth lands)
    - tags        = ["codeorch", f"stage:{stage}"] for filterable analytics
    - trace_name  = "codeorch.run" — human-readable, filterable

Sessions docs:           https://langfuse.com/docs/observability/features/sessions
Instrumentation docs:    https://langfuse.com/docs/observability/sdk/python/instrumentation
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from functools import lru_cache
from typing import Any, Iterator
from uuid import UUID

from langfuse import Langfuse, get_client, propagate_attributes


@lru_cache(maxsize=1)
def get_langfuse() -> Langfuse:
    """Return a singleton Langfuse client. Reads env at first call —
    must be invoked AFTER load_dotenv()."""
    return Langfuse(
        public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
        secret_key=os.environ["LANGFUSE_SECRET_KEY"],
        host=os.environ.get("LANGFUSE_BASE_URL", "https://cloud.langfuse.com"),
    )


class _SpanProxy:
    """Adapter so callers can use a stable .update() shape regardless of
    whether the underlying observation is a span or a generation."""

    def __init__(self, span: Any):
        self._span = span

    def update(
        self,
        output: Any = None,
        metadata: dict[str, Any] | None = None,
        level: str | None = None,
        status_message: str | None = None,
        usage_details: dict[str, int] | None = None,
        model: str | None = None,
    ) -> None:
        kwargs: dict[str, Any] = {}
        if output is not None:
            kwargs["output"] = output
        if metadata is not None:
            kwargs["metadata"] = metadata
        if level is not None:
            kwargs["level"] = level
        if status_message is not None:
            kwargs["status_message"] = status_message
        if usage_details is not None:
            kwargs["usage_details"] = usage_details
        if model is not None:
            kwargs["model"] = model
        if kwargs:
            self._span.update(**kwargs)


@contextmanager
def trace_run(
    run_id: UUID,
    spec: str,
    user_id: str = "anon",
    extra_tags: list[str] | None = None,
) -> Iterator[_SpanProxy]:
    """Open the OUTER span for one CodeOrch run.

    Every agent span opened inside this context becomes a child of this
    span and inherits session_id (= run_id) so the run shows up as a
    single session in the Langfuse Sessions view.
    """
    lf = get_langfuse()
    tags = ["codeorch"] + (extra_tags or [])
    with lf.start_as_current_observation(
        as_type="span",
        name="codeorch.run",
        input={"spec": spec},
    ) as root:
        with propagate_attributes(
            session_id=str(run_id),
            user_id=user_id,
            tags=tags,
            trace_name="codeorch.run",
        ):
            yield _SpanProxy(root)
        # flush is best-effort here — base_agent + smoke test also call
        # flush, but exit-time flush is the safety net for short scripts.
        lf.flush()


@contextmanager
def trace_agent(
    agent_name: str,
    stage: str,
    model: str,
    input_data: dict[str, Any] | None = None,
    as_type: str = "span",
) -> Iterator[_SpanProxy]:
    """Open a CHILD span for one agent call inside an active run trace.

    Pass as_type='generation' when the span wraps a single LLM call —
    Langfuse will compute cost from token usage automatically.
    """
    lf = get_client()
    with lf.start_as_current_observation(
        as_type=as_type,
        name=f"agent.{agent_name}",
        input=input_data,
        metadata={"stage": stage, "model": model},
    ) as span:
        yield _SpanProxy(span)
