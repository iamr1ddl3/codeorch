"""Day 6 e2e smoke — full pipeline with Reviewer + Documenter.

Verifies all 7 stages persist on a happy path:
    spec -> plan -> code -> tests -> gate -> review -> doc

(Plus orchestrator stage = 8 total records).
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

SPEC = "Write a Python function `clamp(x: float, low: float, high: float) -> float` that bounds x. Include 2 pytest tests."

EXPECTED_STAGES = ["spec", "plan", "code", "tests", "gate", "review", "doc", "orchestrator"]


def main() -> int:
    for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "POSTGRES_URL"):
        assert os.environ.get(k), f"{k} missing"

    store = ContextStore()
    run_id = uuid4()
    print(f"run_id = {run_id}")

    started = time.monotonic()
    with trace_run(run_id=run_id, spec=SPEC, user_id="day6-smoke",
                    extra_tags=["day-6", "e2e"]) as root:
        # Mirror what the API does: write the 'spec' stage first.
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
    missing = set(EXPECTED_STAGES) - stages_seen
    assert not missing, f"missing stages: {missing}; got {stages_seen}"
    print(f"  [ok] all {len(EXPECTED_STAGES)} expected stages persisted: "
          f"{sorted(stages_seen)}")

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

    print(f"\n  Final API deliverable preview (first 200 chars):")
    print(f"  {(out.get('documented_code') or '')[:200]}...")

    print(f"\nDAY 6 PIPELINE PASSED — {total_ms}ms total")
    print(f"  Inspect Langfuse: trace 'codeorch.run' session={run_id}")
    print(f"  All 7 nested agent spans + Orchestrator summary present.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
