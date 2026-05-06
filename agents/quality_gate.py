"""Quality Gate — Haiku 4.5, LLM-as-judge.

Production scoring path. As of Day 7 RouterBench, this uses the
verdict-only prompt (`GATE_SYSTEM_VERDICT`) by default — RouterBench
Day 7 (n=39) showed Haiku-verdict at 0.915 mean score / 1.9s vs.
Haiku-weighted at 0.903 / 7.4s. The decomposed rubric was costing
latency without buying score quality.

The weighted rubric is preserved as `GATE_SYSTEM_WEIGHTED` and is the
prompt used by `agents.gate_variants.GateHaikuWeighted` — used in
RouterBench A/B and available for debug-mode scoring on retry/escalate
paths where decomposition aids root-cause analysis.

Routing decision derived from the score:
    score >= 0.75 -> verdict='pass',     routing='reviewer'
    0.50 <= score < 0.75 -> verdict='retry',    routing='retry'
    score <  0.50 -> verdict='escalate', routing='orchestrator'

The 'issues' array supports SLI #5 (hallucination flag rate). Anything the
gate flags as a hallucination (made-up API, wrong type signature, fabricated
import) goes there.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from agents.anthropic_client import call_model
from agents.base_agent import BaseAgent
from agents.models import ROUTING
from observability.langfuse import trace_agent

GATE_SYSTEM_VERDICT = """You are the Quality Gate. Score a code-generation run on a 0.0-1.0 scale.

You receive (a) the plan with acceptance criteria, (b) the Coder's
implementation files, and (c) the Tester's test files. Score the run
holistically on correctness, test coverage, code quality, completeness.

Output ONLY valid JSON, no prose, no markdown fences:

{
  "score": 0.83,
  "verdict": "pass" | "retry" | "escalate",
  "routing": "reviewer" | "retry" | "orchestrator",
  "issues": ["specific issue 1", ...]
}

Routing rules:
- score >= 0.75 -> verdict="pass",     routing="reviewer"
- 0.50 <= score < 0.75 -> verdict="retry",    routing="retry"
- score  < 0.50 -> verdict="escalate", routing="orchestrator"

Set verdict + routing CONSISTENT with score. Flag hallucinations
(non-existent APIs, fabricated imports, wrong type signatures) in issues.
"""

# Weighted rubric prompt — preserved for RouterBench A/B and as a debug-mode
# gate on retry/escalate verdicts. Not the production default as of Day 7.
GATE_SYSTEM_WEIGHTED = """You are the Quality Gate. You are an LLM-as-judge that scores the output
of a multi-agent code generation pipeline. You receive (a) the original plan,
(b) the Coder's implementation files, and (c) the Tester's test suite.

Score 0.0 - 1.0 using a WEIGHTED rubric. Compute each component, then a
weighted total. Show the math in your reasoning.

Rubric:
- correctness   (weight 0.4) — does the code implement the plan? does it
                                pass the test suite when mentally executed?
- test coverage (weight 0.3) — do tests cover every acceptance_criterion?
                                are edge cases tested?
- code quality  (weight 0.2) — readability, idiomatic style, no smells.
- completeness  (weight 0.1) — files complete, no TODOs, dependencies listed.

Output ONLY valid JSON, no prose, no markdown fences:

{
  "score": 0.83,
  "verdict": "pass" | "retry" | "escalate",
  "routing": "reviewer" | "retry" | "orchestrator",
  "issues": ["specific issue 1", "specific issue 2"],
  "components": {
    "correctness":   {"score": 0.9, "weighted": 0.36, "note": "..."},
    "test_coverage": {"score": 0.8, "weighted": 0.24, "note": "..."},
    "code_quality":  {"score": 0.85, "weighted": 0.17, "note": "..."},
    "completeness":  {"score": 0.6, "weighted": 0.06, "note": "..."}
  }
}

Routing rules:
- score >= 0.75 -> verdict="pass",     routing="reviewer"
- 0.50 <= score < 0.75 -> verdict="retry",    routing="retry"
- score  < 0.50 -> verdict="escalate", routing="orchestrator"

Set verdict and routing CONSISTENT with the score per the rules above.
Flag hallucinations explicitly in "issues" — anything the Coder fabricated
(non-existent APIs, wrong signatures, made-up imports).
"""

# Backwards-compat alias for code that still imports GATE_SYSTEM.
GATE_SYSTEM = GATE_SYSTEM_VERDICT


class QualityGate(BaseAgent):
    stage = "gate"
    agent_name = "QualityGate"
    model = ROUTING["quality_gate"]["model"]

    def execute(self, run_id: UUID, inputs: dict[str, Any]) -> dict[str, Any]:
        plan = inputs.get("plan")
        code = inputs.get("code")
        tests = inputs.get("tests")
        if not (plan and code and tests):
            raise ValueError("QualityGate: missing one of plan/code/tests")

        # Compact payload for the gate. Plan/code/tests structures contain
        # full file content which inflates input tokens. Gate only needs
        # task ids + acceptance criteria + the actual code/tests strings.
        compact_plan = {
            "tasks": [
                {"id": t.get("id"), "ac": t.get("acceptance_criterion")}
                for t in plan.get("tasks", [])
            ],
            "acceptance_criteria": plan.get("acceptance_criteria", []),
        }
        payload = {
            "plan": compact_plan,
            "code_files": code.get("files", {}),
            "test_files": tests.get("test_files", {}),
        }
        with trace_agent(
            agent_name=f"{self.agent_name}.llm",
            stage=self.stage,
            model=self.model,
            input_data={
                "plan_tasks": len(plan.get("tasks", [])),
                "code_files": len(code.get("files", {})),
                "test_files": len(tests.get("test_files", {})),
            },
            as_type="generation",
        ) as gen:
            call = call_model(
                model=self.model,
                system=GATE_SYSTEM_VERDICT,
                user=json.dumps(payload),  # no indent — saves ~30% tokens
                max_tokens=512,  # verdict-only output is small; capped for Haiku speed
            )
            gen.update(
                output=call.parsed,
                model=call.model,
                usage_details=call.usage,
                metadata={"latency_ms": call.latency_ms},
            )

        out = call.parsed or {}
        # Capture the actual LLM latency on the output_json — independent of
        # DB write or Langfuse flush time, both of which are measured elsewhere.
        out["llm_latency_ms"] = call.latency_ms
        score = out.get("score")
        if not isinstance(score, (int, float)) or not 0.0 <= score <= 1.0:
            raise ValueError(f"QualityGate: invalid score {score!r}")

        # Enforce verdict/routing consistency in case the model contradicts itself.
        if score >= 0.75:
            out["verdict"], out["routing"] = "pass", "reviewer"
        elif score >= 0.50:
            out["verdict"], out["routing"] = "retry", "retry"
        else:
            out["verdict"], out["routing"] = "escalate", "orchestrator"

        # Persist the score on the row so the dashboard can read it directly.
        # base_agent.run() will write the full output_json; we set self._score
        # so a future override could pass it through to write_stage.
        return out
