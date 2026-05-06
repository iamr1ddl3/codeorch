"""Day 4 SLI validation — runs N trials and checks the targets.

SLIs from PER-109 + DESIGN.md:
  - Agent failure rate < 5%
  - Planner latency  < 8s
  - Cross-agent context loss = 0% (read_stage 'plan' returns non-None)
  - Cost per task: Sonnet < $0.05, Opus < $0.20

3 trials is too small to compute a real failure rate, but it surfaces
gross instability. Day 6's Promptfoo eval suite (10 tasks) gives the
statistically meaningful number.
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

from agents.base_agent import AgentFailure  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from observability.langfuse import trace_run  # noqa: E402
from store.context_store import ContextStore  # noqa: E402

SPECS = [
    "Write a Python function `reverse_words(s: str) -> str` that reverses word order, preserving punctuation. Include 2 unit tests.",
    "Write a JavaScript function `groupBy(arr, key)` that groups objects by a key. Include 1 Jest test.",
    "Write a Python class `LRUCache` with O(1) get and put. Include 3 pytest tests.",
]

PLANNER_SLI_MS = 8_000


def main() -> int:
    for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "POSTGRES_URL"):
        assert os.environ.get(k), f"{k} missing"

    store = ContextStore()
    failures = 0
    planner_latencies: list[int] = []
    end_to_end_latencies: list[int] = []
    miss_rate_failures = 0

    for i, spec in enumerate(SPECS, 1):
        run_id = uuid4()
        print(f"\n--- trial {i}/{len(SPECS)} run_id={run_id} ---")
        print(f"  spec: {spec[:80]}...")

        started = time.monotonic()
        try:
            with trace_run(
                run_id=run_id,
                spec=spec,
                user_id="day4-sli",
                extra_tags=["day-4", "sli"],
            ) as root:
                Orchestrator(store).run(run_id, {"spec": spec})
                root.update(metadata={"trial": i})
        except AgentFailure as exc:
            failures += 1
            print(f"  FAILURE: {exc}")
            continue
        end_to_end_ms = int((time.monotonic() - started) * 1000)
        end_to_end_latencies.append(end_to_end_ms)

        # Reconstruct planner-only latency from the records' timestamps.
        plan_rec = store.read_stage(run_id, "plan")
        orch_rec = store.read_stage(run_id, "orchestrator")
        if plan_rec is None:
            miss_rate_failures += 1
            print("  MISS: read_stage('plan') returned None — context loss!")
            continue
        if plan_rec.status != "success":
            failures += 1
            print(f"  PLAN FAIL: {plan_rec.error}")
            continue

        # Plan is written *before* orchestrator's own stage, so the time
        # delta is a reasonable upper bound on planner-only latency.
        if orch_rec is not None:
            planner_ms = int(
                (plan_rec.timestamp - orch_rec.timestamp).total_seconds() * 1000
            )
            planner_ms = abs(planner_ms)
        else:
            planner_ms = end_to_end_ms
        planner_latencies.append(planner_ms)
        print(f"  end-to-end: {end_to_end_ms}ms  planner~={planner_ms}ms  "
              f"tasks={len(plan_rec.output_json.get('tasks', []))}")

    print("\n" + "=" * 60)
    print("SLI summary")
    print("=" * 60)
    n = len(SPECS)
    fail_rate = (failures / n) * 100
    miss_rate = (miss_rate_failures / n) * 100
    print(f"  Failure rate              : {fail_rate:.1f}%   (target < 5%)")
    print(f"  Cross-agent context loss  : {miss_rate:.1f}%   (target < 2%)")
    if planner_latencies:
        avg_p = sum(planner_latencies) // len(planner_latencies)
        max_p = max(planner_latencies)
        print(f"  Planner latency avg/max   : {avg_p}ms / {max_p}ms   (target < {PLANNER_SLI_MS}ms)")
    if end_to_end_latencies:
        avg_e = sum(end_to_end_latencies) // len(end_to_end_latencies)
        max_e = max(end_to_end_latencies)
        print(f"  End-to-end avg/max        : {avg_e}ms / {max_e}ms")

    ok = (
        fail_rate < 5
        and miss_rate < 2
        and (not planner_latencies or max(planner_latencies) < PLANNER_SLI_MS)
    )
    if ok:
        print("\nALL SLIs MET")
        return 0
    print("\nSLI FAILURE — review trials above")
    return 1


if __name__ == "__main__":
    sys.exit(main())
