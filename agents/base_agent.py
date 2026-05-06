"""Base class for all CodeOrch agents.

Implements two of Dhruv's core patterns:
  - failure isolation: any exception writes a partial-state record with
    status='failure' so the orchestrator can inspect, retry, or escalate.
  - stateless execution: agents read prior stage outputs from the context
    store via run() — they never receive in-memory objects from peers.

Tracing: each agent call opens a child span inside the run's outer trace.
The outer trace is opened by the API or smoke test via trace_run().
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any
from uuid import UUID

from observability.langfuse import trace_agent
from store.context_store import ContextStore


class AgentFailure(Exception):
    """Raised when an agent cannot produce output. Caught by run() and
    persisted as a partial-state record."""


class BaseAgent(ABC):
    stage: str
    agent_name: str
    model: str

    def __init__(self, store: ContextStore | None = None):
        self.store = store or ContextStore()

    def execute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        """Sync execute. Override in sync agents (Orchestrator, Planner)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either execute() or aexecute()"
        )

    async def aexecute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async execute. Override in agents that fan out via asyncio.gather
        (Coder, Tester)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement either execute() or aexecute()"
        )

    def run(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        """Sync entrypoint. Wraps execute() in tracing + try/except."""
        return self._run_with_handler(run_id, inputs, sync=True)

    async def arun(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        """Async entrypoint. Mirrors run() but awaits aexecute().

        Used inside asyncio.gather() for Day 5's parallel Coder + Tester
        fan-out — true parallelism, not sequential.
        """
        started_at = time.monotonic()
        with trace_agent(
            agent_name=self.agent_name,
            stage=self.stage,
            model=self.model,
            input_data=inputs,
        ) as span:
            try:
                output = await self.aexecute(run_id, inputs)
                latency_ms = int((time.monotonic() - started_at) * 1000)
                self.store.write_stage(
                    run_id=run_id,
                    stage=self.stage,
                    agent_name=self.agent_name,
                    output_json=output,
                    status="success",
                )
                span.update(output=output, metadata={"latency_ms": latency_ms})
                return output
            except Exception as exc:
                self._persist_failure(run_id, exc, started_at, span)
                raise AgentFailure(str(exc)) from exc

    def _run_with_handler(
        self, run_id: UUID, inputs: dict[str, Any], sync: bool
    ) -> dict[str, Any]:
        started_at = time.monotonic()
        with trace_agent(
            agent_name=self.agent_name,
            stage=self.stage,
            model=self.model,
            input_data=inputs,
        ) as span:
            try:
                output = self.execute(run_id, inputs)
                latency_ms = int((time.monotonic() - started_at) * 1000)
                self.store.write_stage(
                    run_id=run_id,
                    stage=self.stage,
                    agent_name=self.agent_name,
                    output_json=output,
                    status="success",
                )
                span.update(output=output, metadata={"latency_ms": latency_ms})
                return output
            except Exception as exc:
                self._persist_failure(run_id, exc, started_at, span)
                raise AgentFailure(str(exc)) from exc

    def _persist_failure(
        self, run_id: UUID, exc: Exception, started_at: float, span: Any
    ) -> None:
        latency_ms = int((time.monotonic() - started_at) * 1000)
        self.store.write_stage(
            run_id=run_id,
            stage=self.stage,
            agent_name=self.agent_name,
            output_json={"partial": True},
            status="failure",
            error=f"{type(exc).__name__}: {exc}",
        )
        span.update(
            level="ERROR",
            status_message=f"{type(exc).__name__}: {exc}",
            metadata={"latency_ms": latency_ms},
        )
