"""Deterministic unit tests for the adaptive self-correction loop.

No network, no Postgres, no Langfuse. We drive Orchestrator.execute() with
stubbed agents that script the gate's score trajectory, and assert:

  - retry context (previous_attempt + gate_issues) actually reaches the Coder
    (the latent bug: coder.py used to drop it),
  - the loop converges and returns the BEST-scoring attempt,
  - a regressing or plateauing retry stops the loop early (adaptive stop),
  - best-attempt retention keeps the higher-scoring earlier code,
  - a mid-loop Coder failure returns the best attempt so far.

Strategy: subclass Orchestrator, replace the six agent objects with stubs in
__init__, and override _summarize() so we don't touch the summary LLM or
Langfuse. The stubs override run()/arun() directly so base_agent's tracing +
store writes are bypassed entirely.

Run:  python tests/test_self_correction.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID, uuid4

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from agents.base_agent import AgentFailure  # noqa: E402
from agents.orchestrator import MAX_RETRIES, Orchestrator  # noqa: E402

PLAN = {
    "tasks": [{"id": "T1", "acceptance_criterion": "clamp(x, lo, hi)"}],
    "acceptance_criteria": ["clamp works"],
}
TESTS = {"test_files": {"test_clamp.py": "def test_clamp(): ..."}}


def _code(tag: str) -> dict:
    """A Coder-shaped output whose files content is tagged so we can assert
    WHICH attempt's code was retained."""
    return {
        "files": {"clamp.py": f"# attempt={tag}\n"},
        "language": "python",
        "dependencies": [],
        "notes": tag,
    }


class StubPlanner:
    def run(self, run_id, inputs):
        return PLAN


class StubTester:
    async def arun(self, run_id, inputs):
        return TESTS


class StubCoder:
    """Returns a distinct tagged output per call and records the inputs it
    received, so tests can assert retry context was passed through."""

    def __init__(self):
        self.calls: list[dict] = []

    async def arun(self, run_id, inputs):
        self.calls.append(inputs)
        # First pass has no retry_attempt; retries are 1, 2, ...
        tag = f"attempt-{inputs.get('retry_attempt', 0)}"
        return _code(tag)


class BoomOnRetryCoder(StubCoder):
    async def arun(self, run_id, inputs):
        self.calls.append(inputs)
        if inputs.get("retry_attempt", 0) >= 1:
            raise AgentFailure("intentional retry failure")
        return _code("attempt-0")


class ScriptedGate:
    """Returns a scripted score per call. Verdict/routing derived exactly like
    the real QualityGate (>=0.75 pass, >=0.50 retry, else escalate)."""

    def __init__(self, scores: list[float]):
        self.scores = scores
        self.i = 0

    def run(self, run_id, inputs):
        score = self.scores[min(self.i, len(self.scores) - 1)]
        self.i += 1
        if score >= 0.75:
            verdict, routing = "pass", "reviewer"
        elif score >= 0.50:
            verdict, routing = "retry", "retry"
        else:
            verdict, routing = "escalate", "orchestrator"
        return {
            "score": score,
            "verdict": verdict,
            "routing": routing,
            "issues": [f"issue@{score}"],
        }


class StubReviewer:
    def run(self, run_id, inputs):
        # Echo the code it was given so we can assert which attempt reached it.
        return {"approved": True, "notes": [], "final_code": inputs["code"]}


class StubDocumenter:
    def run(self, run_id, inputs):
        return {"documented_code": inputs["final_code"], "summary": "docs"}


class HarnessOrchestrator(Orchestrator):
    """Orchestrator with stubbed agents and a no-LLM _summarize override."""

    def __init__(self, coder, gate):
        # Deliberately skip super().__init__ — we don't want a ContextStore or
        # real agents. Set only what execute() touches.
        self.store = None
        self.planner = StubPlanner()
        self.coder = coder
        self.tester = StubTester()
        self.gate = gate
        self.reviewer = StubReviewer()
        self.documenter = StubDocumenter()

    def _summarize(self, run_id, spec, plan, code, tests, gate_out,
                   retries_used, review_out=None, doc_out=None,
                   status="completed"):
        # Mirror the real summary's field shape WITHOUT calling the LLM.
        deliverable = (
            (doc_out or {}).get("documented_code")
            or (review_out or {}).get("final_code")
            or code
        )
        return {
            "code_files": list(code.get("files", {}).keys()),
            "code_notes": code.get("notes"),
            "gate_score": gate_out.get("score"),
            "gate_verdict": gate_out.get("verdict"),
            "retries_used": retries_used,
            "documented_code": deliverable,
            "status": status,
        }


def _run(coder, gate) -> dict:
    return HarnessOrchestrator(coder, gate).execute(uuid4(), {"spec": "clamp"})


def test_context_passthrough():
    """Proves the latent bug is dead: retry calls carry previous_attempt +
    gate_issues; the first pass does not."""
    coder = StubCoder()
    out = _run(coder, ScriptedGate([0.60, 0.80]))  # one retry, then pass
    first, retry = coder.calls[0], coder.calls[1]
    assert "previous_attempt" not in first, "first pass must be plan-only"
    assert retry.get("previous_attempt"), "retry must carry previous_attempt"
    assert retry.get("gate_issues"), "retry must carry gate_issues"
    assert retry["retry_attempt"] == 1
    assert out["retries_used"] == 1
    print("[ok] context passthrough — retry carries previous_attempt + gate_issues")


def test_forced_improvement():
    """Ascending scores → loop runs to the ceiling, returns the best (last)."""
    coder = StubCoder()
    out = _run(coder, ScriptedGate([0.60, 0.72, 0.80]))
    assert out["retries_used"] == MAX_RETRIES == 2
    assert out["gate_score"] == 0.80
    assert out["gate_verdict"] == "pass"
    assert out["code_notes"] == "attempt-2"
    print("[ok] forced improvement — converges to best score 0.80 at retry 2")


def test_best_retention_on_regression():
    """A retry that scores LOWER is discarded; first attempt is retained and
    the loop stops early."""
    coder = StubCoder()
    out = _run(coder, ScriptedGate([0.65, 0.55]))
    assert out["retries_used"] == 1, "should stop after the regressing retry"
    assert out["gate_score"] == 0.65, "best (first) score retained"
    assert out["code_notes"] == "attempt-0", "first attempt's code retained, not the regression"
    print("[ok] best retention — regression discarded, first attempt kept, stopped early")


def test_plateau_stops_early():
    """A tie counts as no-improvement → adaptive stop after one retry, proving
    the loop no longer burns the full fixed-2."""
    coder = StubCoder()
    out = _run(coder, ScriptedGate([0.60, 0.60]))
    assert out["retries_used"] == 1, "tie must stop the loop early"
    assert out["gate_score"] == 0.60
    assert out["code_notes"] == "attempt-0", "tie keeps the earlier, cheaper attempt"
    print("[ok] plateau — tie stops early (1 retry, not 2)")


def test_midloop_failure_returns_best():
    """Coder raises on the retry → return the best successful attempt so far."""
    coder = BoomOnRetryCoder()
    out = _run(coder, ScriptedGate([0.60, 0.80]))
    assert out["retries_used"] == 1
    assert out["gate_score"] == 0.60, "best successful attempt retained"
    assert out["code_notes"] == "attempt-0"
    assert out["status"].startswith("retry_failed:")
    print("[ok] mid-loop failure — returns best successful attempt, status=retry_failed")


def test_first_pass_pass_no_retries():
    """Immediate pass → loop body never runs; identical to pre-feature behavior."""
    coder = StubCoder()
    out = _run(coder, ScriptedGate([0.90]))
    assert out["retries_used"] == 0
    assert out["gate_score"] == 0.90
    assert out["code_notes"] == "attempt-0"
    print("[ok] first-pass pass — zero retries, unchanged behavior")


def main():
    tests = [
        test_context_passthrough,
        test_forced_improvement,
        test_best_retention_on_regression,
        test_plateau_stops_early,
        test_midloop_failure_returns_best,
        test_first_pass_pass_no_retries,
    ]
    print("=" * 60)
    print("self-correction loop — deterministic unit tests")
    print("=" * 60)
    for t in tests:
        t()
    print("=" * 60)
    print(f"all {len(tests)} tests passed")


if __name__ == "__main__":
    main()
