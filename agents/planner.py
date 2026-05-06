"""Planner agent — Sonnet 4.

Reads orchestrator stage (which holds the user spec), produces a structured
task breakdown the Coder + Tester will consume in parallel on Day 5.

Output schema (per DESIGN.md):
    {
      "tasks": [
        {"id": str, "description": str, "language": str,
         "complexity": "easy"|"medium"|"hard", "acceptance_criterion": str},
        ...
      ],
      "acceptance_criteria": [str, ...]   # run-level criteria
    }
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from agents.anthropic_client import call_model
from agents.base_agent import BaseAgent
from agents.models import ROUTING
from observability.langfuse import trace_agent

PLANNER_SYSTEM = """You are the Planner agent in a multi-agent code generation pipeline.

Your job: turn a natural-language code spec into a structured task list.
Downstream agents (Coder, Tester) will consume your output IN PARALLEL —
that means each task must be self-contained and unambiguous.

Output ONLY valid JSON matching this schema, no prose, no markdown fences:

{
  "tasks": [
    {
      "id": "T1",
      "description": "concrete what-to-build description",
      "language": "python|typescript|...",
      "complexity": "easy|medium|hard",
      "acceptance_criterion": "single observable check that proves this task is done"
    }
  ],
  "acceptance_criteria": [
    "run-level criterion 1",
    "run-level criterion 2"
  ]
}

Rules:
- 1-5 tasks. More than 5 means you're over-decomposing.
- Each task.id is "T1", "T2", "T3" — sequential.
- complexity: "easy" = under 20 lines, "medium" = 20-80 lines, "hard" = 80+ lines or multi-file.
- acceptance_criterion is testable with a single assertion.
"""


class Planner(BaseAgent):
    stage = "plan"
    agent_name = "Planner"
    model = ROUTING["planner"]["model"]

    def execute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        # Planner reads spec from the orchestrator stage. base_agent has
        # already pre-fetched it and passed it as 'inputs'.
        spec = inputs.get("spec")
        if not spec:
            raise ValueError("Planner: missing 'spec' in inputs")

        # base_agent opens an outer span for this agent. We open a nested
        # generation span so Langfuse computes cost from token usage.
        with trace_agent(
            agent_name=f"{self.agent_name}.llm",
            stage=self.stage,
            model=self.model,
            input_data={"spec": spec},
            as_type="generation",
        ) as gen:
            call = call_model(
                model=self.model,
                system=PLANNER_SYSTEM,
                user=f"Spec:\n{spec}",
                max_tokens=2048,
            )
            gen.update(
                output=call.parsed,
                model=call.model,
                usage_details=call.usage,
                metadata={"latency_ms": call.latency_ms},
            )

        plan = call.parsed or {}
        if not isinstance(plan.get("tasks"), list) or not plan["tasks"]:
            raise ValueError(f"Planner: invalid plan, missing/empty 'tasks': {plan}")
        return plan
