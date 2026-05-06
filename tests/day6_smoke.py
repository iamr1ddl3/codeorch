"""Day 6 e2e smoke — full pipeline with Reviewer + Documenter.

Asserts the contract from agents/orchestrator.py:
    Always: spec -> plan -> code -> tests -> gate -> orchestrator
    Iff gate verdict == 'pass': also review + doc

Both terminal states are valid pipeline outcomes. The smoke verifies the
right stages persist for whichever path was taken.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from agents.orchestrator import Orchestrator  # noqa: E402
from observability.langfuse import trace_run  # noqa: E402
from store.context_store import ContextStore  # noqa: E402

SPEC = "Write a Python function `is_even(n: int) -> bool` that returns True if n is even. Include 2 pytest tests covering one even and one odd input."

CORE_STAGES = ["spec", "plan", "code", "tests", "gate", "orchestrator"]
PASS_ONLY_STAGES = ["review", "doc"]


def main() -> int:
    for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "POSTGRES_URL"):
        assert os.environ.get(k), f"{k} missing"

    store = ContextStore()
    run_id = uuid4()
    print(f"run_id = {run_id}")

    started = time.monotonic()
    with trace_run(run_id=run_id, spec=SPEC, user_id="day6-smoke",
                    extra_tags=["day-6", "e2e"]) as root:
        store.write_stage(run_id, "spec", "api", {"spec": SPEC}, status="success")
        out = Orchestrator(store).run(run_id, {"spec": SPEC})
        root.update(output={
            "verdict": out.get("gate_verdict"),
            "approved": out.get("approved"),
            "documented": bool(out.get("documented_code")),
        })
    total_ms = int((time.monotonic() - started) * 1000)

    records = store.list_run(run_id)
    stages_seen = {r.stage for r in records}
    verdict = out.get("gate_verdict")
    print(f"  gate verdict={verdict} score={out.get('gate_score')} "
          f"retries={out.get('retries_used')}")

    missing_core = set(CORE_STAGES) - stages_seen
    assert not missing_core, f"core stages missing: {missing_core}; got {stages_seen}"
    print(f"  [ok] all {len(CORE_STAGES)} core stages persisted")

    if verdict == "pass":
        missing_pass = set(PASS_ONLY_STAGES) - stages_seen
        assert not missing_pass, (
            f"gate=pass but Reviewer/Documenter stages missing: {missing_pass}"
        )
        review_rec = store.read_stage(run_id, "review")
        assert review_rec.status == "success", f"review failed: {review_rec.error}"
        review = review_rec.output_json or {}
        assert "approved" in review and "final_code" in review, f"bad review: {review}"
        print(f"  [ok] review approved={review['approved']} "
              f"notes={len(review.get('notes', []))}")

        doc_rec = store.read_stage(run_id, "doc")
        assert doc_rec.status == "success", f"documenter failed: {doc_rec.error}"
        doc = doc_rec.output_json or {}
        assert doc.get("documented_code"), "documenter produced no code"
        assert doc.get("summary"), "documenter produced no summary"
        print(f"  [ok] documenter summary: {doc['summary'][:80]}...")
    else:
        # retry-exhausted / escalated path: Reviewer + Documenter MUST be absent
        # by orchestrator design.
        leaked = set(PASS_ONLY_STAGES) & stages_seen
        assert not leaked, (
            f"gate verdict={verdict} but post-gate stages leaked: {leaked}"
        )
        print(f"  [ok] verdict={verdict} short-circuited as designed "
              f"(no review/doc stages, deliverable falls back to flattened code)")

    deliverable = out.get("documented_code") or ""
    assert deliverable, "orchestrator produced no deliverable"
    print(f"\n  Deliverable preview (first 200 chars):")
    print(f"  {deliverable[:200]}...")

    print(f"\nDAY 6 PIPELINE PASSED — {total_ms}ms total")
    print(f"  Inspect Langfuse: trace 'codeorch.run' session={run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
