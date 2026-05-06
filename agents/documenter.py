"""Documenter agent — Haiku 4.5.

Reads the Reviewer's `final_code` and adds docstrings + inline comments.
Non-blocking on failure (per DESIGN.md): if Documenter raises, the
orchestrator returns the undocumented code rather than escalating.

Output schema (per DESIGN.md):
    {
      "documented_code": str,
      "summary": str           # 1-2 sentence what-this-code-does
    }
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from agents.anthropic_client import call_model
from agents.base_agent import BaseAgent
from agents.models import ROUTING
from observability.langfuse import trace_agent

DOCUMENTER_SYSTEM = """You are the Documenter agent. You receive code that has already been
reviewed and approved. Your job: add docstrings and inline comments without
changing behavior.

Output ONLY valid JSON, no prose, no markdown fences:

{
  "documented_code": "the same code, with docstrings + comments added",
  "summary": "1-2 sentence description of what this code does, for the run summary"
}

Rules:
- Do NOT change code logic. Only add docstrings and comments.
- Use the language's idiomatic docstring style (Google for Python, JSDoc
  for TypeScript/JavaScript, etc.).
- Keep comments useful — explain WHY, not WHAT (the code says what).
- Preserve all `# === path/to/file.ext ===` boundary comments unchanged.
- Summary is for the API response and run dashboard — be specific.
"""


class Documenter(BaseAgent):
    stage = "doc"
    agent_name = "Documenter"
    model = ROUTING["documenter"]["model"]

    def execute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        final_code = inputs.get("final_code")
        if not final_code:
            raise ValueError("Documenter: missing 'final_code' in inputs")

        with trace_agent(
            agent_name=f"{self.agent_name}.llm",
            stage=self.stage,
            model=self.model,
            input_data={"code_chars": len(final_code)},
            as_type="generation",
        ) as gen:
            call = call_model(
                model=self.model,
                system=DOCUMENTER_SYSTEM,
                user=json.dumps({"final_code": final_code}),
                max_tokens=4096,
            )
            gen.update(
                output=call.parsed,
                model=call.model,
                usage_details=call.usage,
                metadata={"latency_ms": call.latency_ms},
            )

        out = call.parsed or {}
        out["llm_latency_ms"] = call.latency_ms
        if not out.get("documented_code"):
            raise ValueError(f"Documenter: missing 'documented_code': {out}")
        return out
