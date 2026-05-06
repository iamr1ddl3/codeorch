"""RouterBench benchmark runner.

Pulls every run from pgvector that has plan + code + tests successfully
written, then re-runs each of the 3 gate variants against that fixed
input. Writes per-(run, variant) rows to routerbench/results.jsonl.

Headline experiment: cost-quality curve at the gate stage.
    haiku-weighted   — production default
    haiku-verdict    — cheaper, faster, less defensible
    sonnet-verdict   — pricier, smarter scoring

Why replay existing runs vs fresh /generate calls:
    The gate is the only stage being A/B-tested. Re-running upstream
    (planner/coder/tester) introduces noise we don't want in the
    comparison. Same plan + same code + same tests routed to 3 gates
    isolates the variable.

Cost: ~30 LLM calls × ~$0.005-0.015 each ≈ $0.20-0.40 total.
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID, uuid4

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

import psycopg  # noqa: E402

from agents.gate_variants import VARIANTS  # noqa: E402
from observability.langfuse import trace_run  # noqa: E402
from store.context_store import ContextStore  # noqa: E402

# Pricing — USD per million tokens. Same table as the dashboard.
PRICING = {
    "claude-opus-4-7":            {"in": 15.00, "out": 75.00},
    "claude-sonnet-4-6":          {"in":  3.00, "out": 15.00},
    "claude-haiku-4-5-20251001":  {"in":  0.80, "out":  4.00},
}


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model)
    if p is None:
        return 0.0
    return (input_tokens / 1e6) * p["in"] + (output_tokens / 1e6) * p["out"]


def fetch_replay_inputs() -> list[dict]:
    """Pull every run from pgvector with all of plan+code+tests successful.

    Returns a list of {run_id, plan, code, tests, spec} dicts. Spec is pulled
    from the orchestrator stage's output_json when present, else None.
    """
    dsn = os.environ["POSTGRES_URL"]
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT run_id, stage, output_json
            FROM agent_context
            WHERE status = 'success' AND stage IN ('plan','code','tests','orchestrator','spec')
              AND run_id IN (
                  SELECT run_id FROM agent_context
                  WHERE stage IN ('plan','code','tests') AND status='success'
                  GROUP BY run_id HAVING COUNT(DISTINCT stage) = 3
              )
            ORDER BY run_id, stage
            """
        )
        rows = cur.fetchall()

    by_run: dict[str, dict] = {}
    for run_id, stage, output_json in rows:
        rec = by_run.setdefault(str(run_id), {})
        rec["run_id"] = str(run_id)
        rec[stage] = output_json
    inputs = []
    for run_id, rec in by_run.items():
        if not all(k in rec for k in ("plan", "code", "tests")):
            continue
        spec = (rec.get("orchestrator") or {}).get("spec") or (rec.get("spec") or {}).get("spec")
        inputs.append({
            "run_id": run_id,
            "spec": spec,
            "plan": rec["plan"],
            "code": rec["code"],
            "tests": rec["tests"],
        })
    return inputs


def classify_difficulty(spec: str | None, plan: dict) -> str:
    """Best-effort difficulty bucket. Uses task complexity from the plan;
    falls back to 'unknown' if the plan didn't tag tasks."""
    if not plan:
        return "unknown"
    complexities = [
        (t.get("complexity") or "").lower()
        for t in (plan.get("tasks") or [])
    ]
    if any(c == "hard" for c in complexities):
        return "hard"
    if any(c == "medium" for c in complexities):
        return "medium"
    if any(c == "easy" for c in complexities):
        return "easy"
    return "unknown"


def main() -> int:
    for k in ("ANTHROPIC_API_KEY", "LANGFUSE_PUBLIC_KEY", "POSTGRES_URL"):
        assert os.environ.get(k), f"{k} missing"

    inputs = fetch_replay_inputs()
    print(f"loaded {len(inputs)} runs with replayable plan+code+tests")
    if not inputs:
        print("nothing to replay — run some /generate calls first")
        return 1

    out_path = Path(__file__).parent / "results.jsonl"
    store = ContextStore()
    started_total = time.monotonic()
    rows_written = 0

    with out_path.open("w") as f:
        for i, replay in enumerate(inputs, 1):
            difficulty = classify_difficulty(replay.get("spec"), replay["plan"])
            task_count = len(replay["plan"].get("tasks", []))
            print(f"\n[{i}/{len(inputs)}] run_id={replay['run_id']} "
                  f"difficulty={difficulty} tasks={task_count}")
            for variant_name, variant_cls in VARIANTS.items():
                # Each variant gets a fresh run_id so traces don't collide.
                # The original run_id is recorded in metadata for joinable analysis.
                bench_run_id = uuid4()
                started = time.monotonic()
                try:
                    with trace_run(
                        run_id=bench_run_id,
                        spec=f"[routerbench replay of {replay['run_id']}] {replay.get('spec') or ''}",
                        user_id="routerbench",
                        extra_tags=["routerbench", "day-7", f"variant:{variant_name}",
                                    f"difficulty:{difficulty}"],
                    ) as root:
                        out = variant_cls(store).run(
                            bench_run_id,
                            {
                                "plan": replay["plan"],
                                "code": replay["code"],
                                "tests": replay["tests"],
                            },
                        )
                        root.update(
                            output={
                                "score": out.get("score"),
                                "verdict": out.get("verdict"),
                                "variant": variant_name,
                            },
                        )
                    wall_ms = int((time.monotonic() - started) * 1000)
                    row = {
                        "bench_run_id": str(bench_run_id),
                        "source_run_id": replay["run_id"],
                        "variant": variant_name,
                        "model": out.get("model"),
                        "difficulty": difficulty,
                        "task_count": task_count,
                        "score": out.get("score"),
                        "verdict": out.get("verdict"),
                        "issues_count": len(out.get("issues") or []),
                        "input_tokens": out.get("input_tokens", 0),
                        "output_tokens": out.get("output_tokens", 0),
                        "llm_latency_ms": out.get("llm_latency_ms"),
                        "wall_ms": wall_ms,
                        "cost_usd": cost_usd(
                            out.get("model", ""),
                            out.get("input_tokens", 0),
                            out.get("output_tokens", 0),
                        ),
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    print(f"    {variant_name:16s} "
                          f"score={row['score']:.2f} verdict={row['verdict']:8s} "
                          f"latency={row['llm_latency_ms']:>5d}ms "
                          f"cost=${row['cost_usd']:.5f}")
                except Exception as exc:
                    row = {
                        "bench_run_id": str(bench_run_id),
                        "source_run_id": replay["run_id"],
                        "variant": variant_name,
                        "difficulty": difficulty,
                        "error": f"{type(exc).__name__}: {exc}",
                        "ts": datetime.now(timezone.utc).isoformat(),
                    }
                    print(f"    {variant_name:16s} ERROR: {exc}")

                f.write(json.dumps(row) + "\n")
                f.flush()
                rows_written += 1

    total_s = time.monotonic() - started_total
    print(f"\nwrote {rows_written} rows to {out_path}")
    print(f"total wall time: {total_s:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
