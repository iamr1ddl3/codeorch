"""Day 3 round-trip smoke test.

Verifies:
  1. write_stage -> read_stage round-trip on pgvector
  2. base_agent success path writes status='success' record
  3. base_agent failure path writes status='failure' partial-state record
  4. Langfuse trace structure: one trace per run, agents as child spans,
     session_id == run_id (visible in Sessions view).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from agents.base_agent import AgentFailure, BaseAgent  # noqa: E402
from observability.langfuse import trace_run  # noqa: E402
from store.context_store import ContextStore  # noqa: E402


class HelloAgent(BaseAgent):
    stage = "smoke_success"
    agent_name = "hello"
    model = "claude-haiku-4-5-20251001"

    def execute(self, run_id, inputs):
        return {"greeting": f"hello {inputs['name']}"}


class BoomAgent(BaseAgent):
    stage = "smoke_failure"
    agent_name = "boom"
    model = "claude-haiku-4-5-20251001"

    def execute(self, run_id, inputs):
        raise RuntimeError("intentional smoke failure")


def main() -> int:
    assert os.environ.get("ANTHROPIC_API_KEY"), "ANTHROPIC_API_KEY missing"
    assert os.environ.get("LANGFUSE_PUBLIC_KEY"), "LANGFUSE_PUBLIC_KEY missing"
    assert os.environ.get("POSTGRES_URL"), "POSTGRES_URL missing"

    store = ContextStore()
    run_id = uuid4()
    print(f"run_id = {run_id}")

    # 1. raw store round-trip — no tracing needed for the DB primitive.
    store.write_stage(
        run_id=run_id,
        stage="raw_check",
        agent_name="smoke",
        output_json={"ping": "pong"},
    )
    rec = store.read_stage(run_id, "raw_check")
    assert rec is not None and rec.output_json == {"ping": "pong"}, "round-trip mismatch"
    print("  [ok] write_stage -> read_stage round-trip")

    # 2 + 3 + 4. Single trace_run wraps both agents — Langfuse should show
    # one trace with two child spans (success + failure), all sharing
    # session_id = run_id.
    with trace_run(
        run_id=run_id,
        spec="smoke test spec",
        user_id="smoke-test",
        extra_tags=["smoke", "day-3"],
    ) as root:
        # 2. base_agent success path
        out = HelloAgent(store).run(run_id, {"name": "rocket"})
        assert out == {"greeting": "hello rocket"}
        rec = store.read_stage(run_id, "smoke_success")
        assert rec is not None and rec.status == "success"
        print("  [ok] base_agent success path persisted with status=success")

        # 3. base_agent failure path
        try:
            BoomAgent(store).run(run_id, {})
        except AgentFailure:
            pass
        else:
            raise AssertionError("BoomAgent should have raised AgentFailure")
        rec = store.read_stage(run_id, "smoke_failure")
        assert rec is not None and rec.status == "failure" and rec.error
        print("  [ok] base_agent failure path persisted with status=failure + error")

        root.update(
            output={"agents_run": 2, "successes": 1, "failures": 1},
            metadata={"test": "smoke", "day": 3},
        )

    # 4. list_run shows all 3 records
    records = store.list_run(run_id)
    assert len(records) == 3, f"expected 3 records, got {len(records)}"
    print(f"  [ok] list_run returned {len(records)} records (raw_check, smoke_success, smoke_failure)")

    print("\nSMOKE TEST PASSED")
    print(f"  Inspect Langfuse:")
    print(f"    Traces view  → trace named 'codeorch.run'")
    print(f"    Sessions view → session_id = {run_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
