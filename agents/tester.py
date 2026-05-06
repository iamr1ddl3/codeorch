"""Tester agent — Sonnet 4.6, async.

Reads ONLY the plan stage (NOT the Coder's output) — generates the test
suite from the plan's acceptance criteria alone. This independence is
deliberate: it means the Quality Gate compares Coder output against tests
written from the spec, not tests written from the implementation.

Output schema (per DESIGN.md):
    {
      "test_files": {"path/to/test.ext": "content", ...},
      "framework": "pytest|jest|vitest|...",
      "coverage_targets": ["function or behavior covered", ...]
    }
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from agents.anthropic_client import acall_model
from agents.base_agent import BaseAgent
from agents.models import ROUTING
from observability.langfuse import trace_agent

TESTER_SYSTEM = """You are the Tester agent in a multi-agent code generation pipeline.

You receive a structured plan with tasks and acceptance_criteria. Your job:
write a test suite that verifies the acceptance_criteria — WITHOUT seeing
the implementation. Tests must be specification-driven, not implementation-
driven (this is deliberate: the Quality Gate uses your tests to score the
Coder's work objectively).

Output ONLY valid JSON, no prose, no markdown fences:

{
  "test_files": {
    "path/to/test.ext": "complete test file content"
  },
  "framework": "pytest|jest|vitest|junit|...",
  "coverage_targets": [
    "specific function or behavior verified",
    ...
  ]
}

Rules:
- One test_file per source-file-under-test, named conventionally for the framework.
- Cover EVERY acceptance_criterion. Each criterion = at least one assertion.
- Include happy-path AND at least one edge case per task.
- Use the framework idiomatic to the language (pytest for Python, jest for TS/JS, etc.).
- Do NOT generate the implementation files; the Coder handles those.
"""


class Tester(BaseAgent):
    stage = "tests"
    agent_name = "Tester"
    model = ROUTING["tester"]["model"]

    async def aexecute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        plan = inputs.get("plan")
        if not plan or not plan.get("tasks"):
            raise ValueError("Tester: missing or empty 'plan.tasks' in inputs")

        with trace_agent(
            agent_name=f"{self.agent_name}.llm",
            stage=self.stage,
            model=self.model,
            input_data={"task_count": len(plan["tasks"])},
            as_type="generation",
        ) as gen:
            call = await acall_model(
                model=self.model,
                system=TESTER_SYSTEM,
                user=f"Plan:\n{json.dumps(plan, indent=2)}",
                max_tokens=4096,
            )
            gen.update(
                output=call.parsed,
                model=call.model,
                usage_details=call.usage,
                metadata={"latency_ms": call.latency_ms},
            )

        out = call.parsed or {}
        if not isinstance(out.get("test_files"), dict) or not out["test_files"]:
            raise ValueError(f"Tester: invalid output, missing/empty 'test_files': {out}")
        return out
