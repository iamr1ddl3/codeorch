"""Orchestrator agent — Opus 4.7.

Day 4 scope: Orchestrator -> Planner only.
Day 5 scope: full Planner -> asyncio.gather(Coder, Tester) -> Quality Gate
             pipeline with self-correction loop on retry verdicts.

Self-correction loop (per Dhruv's vocabulary):
  - Gate verdict='pass'     -> hand off to Reviewer (Day 6) — for now, return.
  - Gate verdict='retry'    -> re-run Coder with the gate's issues injected
                                into context. Tester output is reused (per
                                DESIGN.md "On retry: re-runs Coder only").
                                Max 2 retries.
  - Gate verdict='escalate' -> stop the run, surface escalation_reason.

This is "self-correction loops" + "failure isolation — no cascading
failures" in motion: a failed Coder doesn't kill Tester (asyncio.gather
returns_exceptions=True), and a low-score gate triggers context-injected
retry, not a re-plan from scratch.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from uuid import UUID

from agents.anthropic_client import call_model
from agents.base_agent import AgentFailure, BaseAgent
from agents.coder import Coder
from agents.documenter import Documenter
from agents.models import ROUTING
from agents.planner import Planner
from agents.quality_gate import QualityGate
from agents.reviewer import Reviewer
from agents.tester import Tester
from observability.langfuse import trace_agent
from store.context_store import ContextStore

MAX_RETRIES = 2

ORCHESTRATOR_SUMMARY_SYSTEM = """You are the Orchestrator. You just finished a CodeOrch run. Summarize.

Output ONLY valid JSON, no prose:

{
  "summary": "1-sentence outcome description",
  "verdict": "pass|retry_exhausted|escalated|failed",
  "next_step_for_user": "what they should do with this output"
}
"""


class Orchestrator(BaseAgent):
    stage = "orchestrator"
    agent_name = "Orchestrator"
    model = ROUTING["orchestrator"]["model"]

    def __init__(self, store: ContextStore | None = None):
        super().__init__(store)
        self.planner = Planner(self.store)
        self.coder = Coder(self.store)
        self.tester = Tester(self.store)
        self.gate = QualityGate(self.store)
        self.reviewer = Reviewer(self.store)
        self.documenter = Documenter(self.store)

    def execute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        spec = inputs.get("spec")
        if not spec:
            raise ValueError("Orchestrator: missing 'spec' in inputs")

        # 1. Planner — sync, sequential. Failure aborts the run.
        try:
            plan = self.planner.run(run_id, {"spec": spec})
        except AgentFailure as exc:
            raise AgentFailure(f"Planner failed: {exc}") from exc

        # 2. Coder + Tester in parallel via asyncio.gather. Tester reads only
        # the plan; Coder reads only the plan. Independent inputs, parallel
        # execution. asyncio.gather propagates the first exception by default,
        # but we use return_exceptions=True so a Coder failure doesn't kill
        # the Tester (failure isolation).
        code, tests = self._fanout_coder_tester(run_id, plan)

        # 3. Quality Gate — sync. Scores code+tests on weighted rubric.
        gate_input = {"plan": plan, "code": code, "tests": tests}
        gate_out = self.gate.run(run_id, gate_input)

        retries_used = 0
        while gate_out.get("verdict") == "retry" and retries_used < MAX_RETRIES:
            retries_used += 1
            # Self-correction loop: re-run Coder ONLY (Tester output is reused
            # per DESIGN.md). Inject the gate's issues into the Coder's context
            # so the model knows what to fix.
            retry_inputs = {
                "plan": plan,
                "previous_attempt": code,
                "gate_issues": gate_out.get("issues", []),
                "retry_attempt": retries_used,
            }
            try:
                code = asyncio.run(self.coder.arun(run_id, retry_inputs))
            except AgentFailure as exc:
                # Failed retry counts as a retry — break out and let gate
                # escalate on the partial-state record.
                return self._summarize(
                    run_id, spec, plan, code, tests, gate_out,
                    retries_used, status=f"retry_failed: {exc}",
                )
            gate_out = self.gate.run(
                run_id,
                {"plan": plan, "code": code, "tests": tests},
            )

        # 4. Reviewer (verdict='pass' only). Skip on retry-exhausted/escalated.
        review_out: dict[str, Any] | None = None
        doc_out: dict[str, Any] | None = None
        if gate_out.get("verdict") == "pass":
            try:
                review_out = self.reviewer.run(
                    run_id, {"code": code, "tests": tests, "gate": gate_out}
                )
            except AgentFailure:
                # Per DESIGN.md: pass-through with approved=false. base_agent
                # already persisted the failure row. Continue so Documenter
                # gets a chance — it can still document the unreviewed code.
                review_out = {"approved": False, "notes": ["review failed"],
                              "final_code": _flatten_code_files(code)}

            # 5. Documenter — non-blocking on failure (per DESIGN.md). If it
            # raises, the orchestrator returns review_out['final_code'] as-is.
            if review_out.get("final_code"):
                try:
                    doc_out = self.documenter.run(
                        run_id, {"final_code": review_out["final_code"]}
                    )
                except AgentFailure:
                    doc_out = None  # explicit: failure record already persisted

        return self._summarize(
            run_id, spec, plan, code, tests, gate_out, retries_used,
            review_out=review_out, doc_out=doc_out,
        )

    def _fanout_coder_tester(
        self, run_id: UUID, plan: dict[str, Any]
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """asyncio.gather Coder + Tester. Re-raises if EITHER fails — but
        the failed agent's partial-state record is already persisted, so the
        Orchestrator's failure record will surface alongside the Coder/Tester
        failure record for postmortem."""
        async def _run_both():
            return await asyncio.gather(
                self.coder.arun(run_id, {"plan": plan}),
                self.tester.arun(run_id, {"plan": plan}),
                return_exceptions=True,
            )

        results = asyncio.run(_run_both())
        code_res, test_res = results
        # Surface the FIRST failure (Coder before Tester) so the orchestrator
        # stage record points to the most actionable error. Both failure rows
        # exist in pgvector regardless.
        if isinstance(code_res, BaseException):
            raise AgentFailure(f"Coder failed in fan-out: {code_res}") from code_res
        if isinstance(test_res, BaseException):
            raise AgentFailure(f"Tester failed in fan-out: {test_res}") from test_res
        return code_res, test_res

    def _summarize(
        self,
        run_id: UUID,
        spec: str,
        plan: dict[str, Any],
        code: dict[str, Any],
        tests: dict[str, Any],
        gate_out: dict[str, Any],
        retries_used: int,
        review_out: dict[str, Any] | None = None,
        doc_out: dict[str, Any] | None = None,
        status: str = "completed",
    ) -> dict[str, Any]:
        # Day 5: Reviewer + Documenter land Day 6. For now, the orchestrator
        # stage is the run summary — the gate verdict is the final outcome.
        with trace_agent(
            agent_name=f"{self.agent_name}.summary",
            stage=self.stage,
            model=self.model,
            input_data={"score": gate_out.get("score"), "verdict": gate_out.get("verdict")},
            as_type="generation",
        ) as gen:
            call = call_model(
                model=self.model,
                system=ORCHESTRATOR_SUMMARY_SYSTEM,
                user=json.dumps({
                    "spec": spec,
                    "plan_task_count": len(plan.get("tasks", [])),
                    "code_files": list(code.get("files", {}).keys()),
                    "test_files": list(tests.get("test_files", {}).keys()),
                    "gate": {
                        "score": gate_out.get("score"),
                        "verdict": gate_out.get("verdict"),
                        "issues": gate_out.get("issues", []),
                    },
                    "retries_used": retries_used,
                }, indent=2),
                max_tokens=512,
            )
            gen.update(
                output=call.parsed,
                model=call.model,
                usage_details=call.usage,
                metadata={"latency_ms": call.latency_ms},
            )

        summary = call.parsed or {}
        # The user-facing deliverable: documented code if available, else
        # reviewed code, else flat-concatenated code files. Always something.
        deliverable = (
            (doc_out or {}).get("documented_code")
            or (review_out or {}).get("final_code")
            or _flatten_code_files(code)
        )
        return {
            "spec": spec,
            "plan_task_count": len(plan.get("tasks", [])),
            "code_files": list(code.get("files", {}).keys()),
            "test_files": list(tests.get("test_files", {}).keys()),
            "gate_score": gate_out.get("score"),
            "gate_verdict": gate_out.get("verdict"),
            "gate_routing": gate_out.get("routing"),
            "gate_issues": gate_out.get("issues", []),
            "retries_used": retries_used,
            "approved": (review_out or {}).get("approved"),
            "review_notes": (review_out or {}).get("notes", []),
            "documented_code": deliverable,
            "doc_summary": (doc_out or {}).get("summary"),
            "summary": summary.get("summary", ""),
            "status": status,
        }


def _flatten_code_files(code: dict[str, Any]) -> str:
    """Concatenate Coder's file dict into a single string with boundary
    comments. Used as a fallback when Reviewer/Documenter fail."""
    parts = []
    for path, content in (code.get("files") or {}).items():
        parts.append(f"# === {path} ===\n{content}")
    return "\n\n".join(parts)
