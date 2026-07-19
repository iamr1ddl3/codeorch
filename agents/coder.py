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

from agents.base_agent import BaseAgent
from agents.models import ROUTING

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

        # Self-correction: on a retry the orchestrator passes the previous
        # attempt + the gate's issues. First pass leaves these absent and the
        # prompt is plan-only (unchanged). CODER_SYSTEM stays fixed — the stable
        # output contract belongs in the system turn; per-attempt critique goes
        # in the user turn.
        previous_attempt = inputs.get("previous_attempt")
        gate_issues = inputs.get("gate_issues") or []
        retry_attempt = inputs.get("retry_attempt", 0)
        is_retry = bool(previous_attempt) and retry_attempt > 0

        if is_retry:
            prev_files = (
                previous_attempt.get("files", {})
                if isinstance(previous_attempt, dict)
                else {}
            )
            issues_block = "\n".join(f"- {i}" for i in gate_issues) or "- (none reported)"
            user = (
                f"Plan:\n{json.dumps(plan, indent=2)}\n\n"
                f"This is RETRY attempt {retry_attempt}. Your previous output scored "
                f"below the quality bar. REVISE the previous files to fix the issues "
                f"below — do NOT start over.\n\n"
                f"Previous attempt files:\n{json.dumps(prev_files, indent=2)}\n\n"
                f"Quality gate issues to fix:\n{issues_block}\n\n"
                f"Return the SAME JSON schema (files/language/dependencies/notes) "
                f"with the issues resolved."
            )
        else:
            user = f"Plan:\n{json.dumps(plan, indent=2)}"

        call = await self._agenerate(
            system=CODER_SYSTEM,
            user=user,
            max_tokens=4096,
            input_data={"task_count": len(plan["tasks"]), "retry_attempt": retry_attempt},
        )

        out = call.parsed or {}
        out["llm_latency_ms"] = call.latency_ms
        if not isinstance(out.get("files"), dict) or not out["files"]:
            raise ValueError(f"Coder: invalid output, missing/empty 'files': {out}")
        return out
