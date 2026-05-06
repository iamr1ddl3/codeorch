"""Coder agent — Sonnet 4.6, async.

Reads the plan stage, implements every task. Runs in parallel with Tester
via asyncio.gather() — fan-out/fan-in execution per Dhruv's vocabulary.

Output schema (per DESIGN.md):
    {
      "files": {"filename.py": "content", ...},
      "language": "python|typescript|...",
      "dependencies": ["pkg==1.0", ...],
      "notes": "implementation rationale"
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

CODER_SYSTEM = """You are the Coder agent in a multi-agent code generation pipeline.

You receive a structured plan with 1-5 tasks. For EACH task, write complete,
runnable code that satisfies its acceptance_criterion. Code must be production
quality — no TODOs, no placeholder stubs.

Output ONLY valid JSON, no prose, no markdown fences:

{
  "files": {
    "path/to/file.ext": "complete file content as a string"
  },
  "language": "python|typescript|javascript|...",
  "dependencies": ["package==version", ...],
  "notes": "1-2 sentence implementation rationale"
}

Rules:
- One key per file. Filenames are relative paths (e.g. "src/debounce.ts").
- "language" is the primary language of the implementation files.
- "dependencies" lists exact pinned packages needed to run the code (omit stdlib).
- Keep files under 200 lines each — break apart if longer.
- Do NOT include test files here; the Tester agent handles those independently.
"""


class Coder(BaseAgent):
    stage = "code"
    agent_name = "Coder"
    model = ROUTING["coder"]["model"]

    async def aexecute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        plan = inputs.get("plan")
        if not plan or not plan.get("tasks"):
            raise ValueError("Coder: missing or empty 'plan.tasks' in inputs")

        with trace_agent(
            agent_name=f"{self.agent_name}.llm",
            stage=self.stage,
            model=self.model,
            input_data={"task_count": len(plan["tasks"])},
            as_type="generation",
        ) as gen:
            call = await acall_model(
                model=self.model,
                system=CODER_SYSTEM,
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
        out["llm_latency_ms"] = call.latency_ms
        if not isinstance(out.get("files"), dict) or not out["files"]:
            raise ValueError(f"Coder: invalid output, missing/empty 'files': {out}")
        return out
