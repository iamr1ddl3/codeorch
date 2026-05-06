"""Reviewer agent — Sonnet 4.6.

Final quality review. Reads code + tests + gate score together and produces
a final approval decision plus the canonical `final_code` block that the
Documenter will document and the API will return.

Output schema (per DESIGN.md):
    {
      "approved": bool,
      "notes": [str, ...],
      "final_code": str           # the canonical code (may = code['files'] joined,
                                  # or with reviewer-applied minor fixes)
    }

On failure: writes `{approved: false, notes: ['review failed']}` —
non-blocking pass-through (per DESIGN.md), so Documenter can still run.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from agents.anthropic_client import call_model
from agents.base_agent import BaseAgent
from agents.models import ROUTING
from observability.langfuse import trace_agent

REVIEWER_SYSTEM = """You are the Reviewer agent — the final human-in-the-loop substitute before
the run completes. You receive (a) the Coder's code files, (b) the Tester's
test files, and (c) the Quality Gate's score and issues.

Your job: hold or release. If the gate passed (score >= 0.75) but you spot a
material issue the gate missed, set approved=false. If the gate flagged
issues but they're cosmetic and the code is correct, you can still approve.

Output ONLY valid JSON, no prose, no markdown fences:

{
  "approved": true,
  "notes": [
    "1-line observation, blocking issue, or commendation",
    ...
  ],
  "final_code": "the canonical code as a single string — concatenated files with file boundary comments"
}

Rules:
- "final_code" is what gets shipped. Concatenate the files in dependency
  order with `# === path/to/file.ext ===` boundary comments between them.
- Do NOT rewrite the code substantively. Minor fixes (typos, missing imports
  the gate flagged) are OK; refactors are not — those route back to retry.
- Notes are 1-line each, max 5. Be specific (file + line if possible).
- If the gate already approved and you have nothing to add, set
  notes=["gate-approved; reviewer concurs"] and approved=true.
"""


class Reviewer(BaseAgent):
    stage = "review"
    agent_name = "Reviewer"
    model = ROUTING["reviewer"]["model"]

    def execute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        code = inputs.get("code")
        tests = inputs.get("tests")
        gate = inputs.get("gate")
        if not (code and tests and gate):
            raise ValueError("Reviewer: missing one of code/tests/gate in inputs")

        # Compact payload — reviewer needs the actual code, not the metadata.
        payload = {
            "code_files": code.get("files", {}),
            "test_files": tests.get("test_files", {}),
            "gate": {
                "score": gate.get("score"),
                "verdict": gate.get("verdict"),
                "issues": gate.get("issues", []),
            },
        }
        with trace_agent(
            agent_name=f"{self.agent_name}.llm",
            stage=self.stage,
            model=self.model,
            input_data={
                "code_files": len(code.get("files", {})),
                "test_files": len(tests.get("test_files", {})),
                "gate_score": gate.get("score"),
            },
            as_type="generation",
        ) as gen:
            call = call_model(
                model=self.model,
                system=REVIEWER_SYSTEM,
                user=json.dumps(payload),
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
        if "approved" not in out or "final_code" not in out:
            raise ValueError(f"Reviewer: missing 'approved' or 'final_code': {out}")
        return out
