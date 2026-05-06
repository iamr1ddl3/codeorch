"""CodeOrch — 8-SLI Operations Dashboard.

Reads from `agent_context` (pgvector) directly. Renders the 8 SLIs from
DESIGN.md / RocketVocabularyIntel.md with thresholds and trend charts.

Run:
    streamlit run dashboard/app.py

This is the headline portfolio artifact. Recruiter screenshots come from
here. Anchor every visual to a Dhruv-vocabulary term.
"""

from __future__ import annotations

import os
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg
import streamlit as st
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env", override=True)

# --- Pricing table — input/output USD per million tokens ---
# Used to compute SLI #6 (cost per task per model). Update if Anthropic
# changes pricing. These are public list prices as of sprint start.
PRICING = {
    "claude-opus-4-7":              {"in": 15.00, "out": 75.00},
    "claude-sonnet-4-6":            {"in":  3.00, "out": 15.00},
    "claude-haiku-4-5-20251001":    {"in":  0.80, "out":  4.00},
}

# SLI thresholds from DESIGN.md / RocketVocabularyIntel.md
SLI = {
    "failure_rate":      {"target": 0.05, "label": "< 5%"},
    "context_loss":      {"target": 0.02, "label": "< 2%"},
    "planner_latency":   {"target": 8_000, "label": "< 8s"},
    "coder_latency":     {"target": 30_000, "label": "< 30s"},
    "gate_latency":      {"target": 5_000, "label": "< 5s"},  # Day 7: verdict-only restored SLI
    "accuracy":          {"target": 0.75, "label": "≥ 75%"},
    "hallucination":     {"target": 0.10, "label": "< 10%"},
    "retry_rate":        {"target": 0.20, "label": "< 20%"},
    "opus_share":        {"target": 0.15, "label": "< 15%"},
    "haiku_share":       {"target": 0.40, "label": "> 40%"},
}

st.set_page_config(
    page_title="CodeOrch — 8 SLI Dashboard",
    page_icon="🚀",
    layout="wide",
)


@st.cache_data(ttl=10)
def load_records(window_hours: int = 24) -> pd.DataFrame:
    """Pull the last `window_hours` of agent_context records into a DataFrame."""
    dsn = os.environ["POSTGRES_URL"]
    cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
    with psycopg.connect(dsn) as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, run_id, stage, agent_name, output_json, score,
                   status, error, timestamp
            FROM agent_context
            WHERE timestamp >= %s
            ORDER BY timestamp ASC
            """,
            (cutoff,),
        )
        rows = cur.fetchall()
    cols = ["id", "run_id", "stage", "agent_name", "output_json", "score",
            "status", "error", "timestamp"]
    df = pd.DataFrame(rows, columns=cols)
    if not df.empty:
        df["run_id"] = df["run_id"].astype(str)
    return df


def _stage_latency_ms(df: pd.DataFrame, stage: str) -> pd.Series:
    """Per-row latency for a given stage, pulled from output_json.llm_latency_ms
    when present, else falls back to wall-time delta from the run's first stage."""
    rows = df[df["stage"] == stage]
    return rows["output_json"].apply(
        lambda o: (o or {}).get("llm_latency_ms") if isinstance(o, dict) else None
    ).dropna()


def _model_for_stage(stage: str) -> str:
    return {
        "orchestrator": "claude-opus-4-7",
        "plan":         "claude-sonnet-4-6",
        "code":         "claude-sonnet-4-6",
        "tests":        "claude-sonnet-4-6",
        "gate":         "claude-haiku-4-5-20251001",
        "review":       "claude-sonnet-4-6",
        "doc":          "claude-haiku-4-5-20251001",
    }.get(stage, "unknown")


def compute_slis(df: pd.DataFrame) -> dict[str, Any]:
    """Reduce the records to one row per (run_id, stage) and compute all SLIs."""
    if df.empty:
        return {"empty": True}

    runs = df["run_id"].nunique()

    # SLI #1 — Agent failure rate. Fraction of agent-stage attempts with status='failure'.
    # Excludes 'spec' (always success — written by API).
    agent_rows = df[df["stage"] != "spec"]
    failure_rate = (agent_rows["status"] == "failure").mean() if not agent_rows.empty else 0.0

    # SLI #2 — Cross-agent context loss. Approximated as: fraction of runs missing
    # at least one expected stage (plan, code, tests, gate). On a fully successful
    # run the orchestrator persists all four.
    expected = {"plan", "code", "tests", "gate"}
    runs_grouped = df.groupby("run_id")["stage"].apply(set)
    context_loss = (runs_grouped.apply(lambda s: bool(expected - s))).mean() if runs > 0 else 0.0

    # SLI #3 — Per-stage latency (avg)
    planner_lat = _stage_latency_ms(df, "plan")
    coder_lat = _stage_latency_ms(df, "code")
    gate_lat = _stage_latency_ms(df, "gate")

    # SLI #4 — Task completion accuracy. Approximated by gate verdict='pass' rate.
    gate_rows = df[df["stage"] == "gate"]
    accuracy = 0.0
    if not gate_rows.empty:
        verdicts = gate_rows["output_json"].apply(
            lambda o: (o or {}).get("verdict") if isinstance(o, dict) else None
        )
        accuracy = (verdicts == "pass").sum() / max(len(verdicts), 1)

    # SLI #5 — Hallucination flags. Fraction of gate runs reporting any 'issues'.
    hallucination = 0.0
    if not gate_rows.empty:
        flagged = gate_rows["output_json"].apply(
            lambda o: bool((o or {}).get("issues")) if isinstance(o, dict) else False
        )
        hallucination = flagged.sum() / max(len(flagged), 1)

    # SLI #6 — Cost per task per model. Need token usage from each generation
    # span. We didn't persist usage on the row; compute from the count of
    # generation calls × an estimate based on stage. (Day 7 RouterBench will
    # rewire to read from Langfuse for exact cost.) For now: approximate.
    cost_by_model: dict[str, float] = defaultdict(float)
    for _, r in df.iterrows():
        if r["status"] != "success" or r["stage"] == "spec":
            continue
        m = _model_for_stage(r["stage"])
        # Rough estimate based on average token usage seen in Day 5 sweeps.
        # Replace with Langfuse-derived cost in Day 7.
        if m == "claude-opus-4-7":
            cost_by_model[m] += 0.05      # orchestrator summary call
        elif m == "claude-sonnet-4-6":
            cost_by_model[m] += 0.015     # planner/coder/tester/reviewer
        elif m == "claude-haiku-4-5-20251001":
            cost_by_model[m] += 0.005     # gate/documenter

    # SLI #7 — Retry rate. Look at orchestrator output_json.retries_used.
    orch_rows = df[df["stage"] == "orchestrator"]
    retry_rate = 0.0
    if not orch_rows.empty:
        retries = orch_rows["output_json"].apply(
            lambda o: (o or {}).get("retries_used", 0) if isinstance(o, dict) else 0
        )
        retry_rate = (retries > 0).sum() / max(len(retries), 1)

    # SLI #8 — Model routing distribution. Count agent calls per model.
    model_counter: Counter = Counter()
    for _, r in df.iterrows():
        if r["stage"] == "spec":
            continue
        model_counter[_model_for_stage(r["stage"])] += 1
    total = sum(model_counter.values()) or 1
    model_share = {m: c / total for m, c in model_counter.items()}

    return {
        "empty": False,
        "runs": runs,
        "records": len(df),
        "failure_rate": float(failure_rate),
        "context_loss": float(context_loss),
        "planner_latency_avg": int(planner_lat.mean()) if len(planner_lat) else 0,
        "planner_latency_p95": int(planner_lat.quantile(0.95)) if len(planner_lat) else 0,
        "coder_latency_avg": int(coder_lat.mean()) if len(coder_lat) else 0,
        "coder_latency_p95": int(coder_lat.quantile(0.95)) if len(coder_lat) else 0,
        "gate_latency_avg": int(gate_lat.mean()) if len(gate_lat) else 0,
        "gate_latency_p95": int(gate_lat.quantile(0.95)) if len(gate_lat) else 0,
        "accuracy": float(accuracy),
        "hallucination": float(hallucination),
        "cost_by_model": dict(cost_by_model),
        "retry_rate": float(retry_rate),
        "model_share": model_share,
        "model_counts": dict(model_counter),
    }


def sli_card(label: str, value: str, target: str, ok: bool, sub: str = ""):
    color = "#1a7f1a" if ok else "#b08000"
    st.markdown(
        f"""<div style="border:1px solid #ccc;border-radius:8px;padding:12px;
        background:#fafafa;height:130px;">
        <div style="font-size:0.85rem;color:#666;">{label}</div>
        <div style="font-size:1.8rem;font-weight:600;color:{color};">{value}</div>
        <div style="font-size:0.75rem;color:#888;">target: {target}</div>
        <div style="font-size:0.7rem;color:#aaa;">{sub}</div>
        </div>""",
        unsafe_allow_html=True,
    )


def main():
    st.title("CodeOrch — 8 SLI Dashboard")
    st.caption(
        "Stateless agents · externalized context store · weighted-rubric quality gate · "
        "multi-model routing — operational view of the pipeline."
    )

    with st.sidebar:
        st.header("Filter")
        window = st.selectbox(
            "Time window",
            options=[1, 6, 24, 168],
            index=2,
            format_func=lambda h: f"last {h}h" if h < 24 else (
                "last 24h" if h == 24 else "last 7d"),
        )
        st.markdown("---")
        if st.button("🔄 Refresh now"):
            st.cache_data.clear()
            st.rerun()

    df = load_records(window_hours=window)
    sli = compute_slis(df)

    if sli.get("empty"):
        st.info("No runs in this window yet. Hit `POST /generate` and refresh.")
        st.stop()

    st.markdown(f"**{sli['runs']} runs · {sli['records']} agent calls** in window.")

    # --- Row 1: failure / context loss / accuracy / retry ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sli_card(
            "1 · Agent failure rate",
            f"{sli['failure_rate']*100:.1f}%",
            SLI["failure_rate"]["label"],
            ok=sli["failure_rate"] < SLI["failure_rate"]["target"],
        )
    with c2:
        sli_card(
            "2 · Cross-agent context loss",
            f"{sli['context_loss']*100:.1f}%",
            SLI["context_loss"]["label"],
            ok=sli["context_loss"] < SLI["context_loss"]["target"],
        )
    with c3:
        sli_card(
            "4 · Task completion (gate=pass)",
            f"{sli['accuracy']*100:.1f}%",
            SLI["accuracy"]["label"],
            ok=sli["accuracy"] >= SLI["accuracy"]["target"],
        )
    with c4:
        sli_card(
            "7 · Retry rate",
            f"{sli['retry_rate']*100:.1f}%",
            SLI["retry_rate"]["label"],
            ok=sli["retry_rate"] < SLI["retry_rate"]["target"],
        )

    # --- Row 2: per-stage latency ---
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        sli_card(
            "3a · Planner latency (avg)",
            f"{sli['planner_latency_avg']/1000:.1f}s",
            SLI["planner_latency"]["label"],
            ok=sli["planner_latency_avg"] < SLI["planner_latency"]["target"],
            sub=f"p95: {sli['planner_latency_p95']/1000:.1f}s",
        )
    with c2:
        sli_card(
            "3b · Coder latency (avg)",
            f"{sli['coder_latency_avg']/1000:.1f}s",
            SLI["coder_latency"]["label"],
            ok=sli["coder_latency_avg"] < SLI["coder_latency"]["target"],
            sub=f"p95: {sli['coder_latency_p95']/1000:.1f}s",
        )
    with c3:
        sli_card(
            "3c · Gate latency (avg)",
            f"{sli['gate_latency_avg']/1000:.1f}s",
            SLI["gate_latency"]["label"],
            ok=sli["gate_latency_avg"] < SLI["gate_latency"]["target"],
            sub="verdict-only since Day 7 RouterBench",
        )
    with c4:
        sli_card(
            "5 · Hallucination flags",
            f"{sli['hallucination']*100:.1f}%",
            SLI["hallucination"]["label"],
            ok=sli["hallucination"] < SLI["hallucination"]["target"],
            sub="gate-flagged issues",
        )

    st.markdown("---")

    # --- Row 3: model routing distribution + cost ---
    cL, cR = st.columns([2, 1])
    with cL:
        st.subheader("8 · Model routing distribution")
        if sli["model_counts"]:
            mdf = pd.DataFrame([
                {"model": m.replace("claude-", "").replace("-20251001", ""),
                 "count": c, "share": sli["model_share"][m]}
                for m, c in sli["model_counts"].items()
            ])
            fig = px.bar(
                mdf, x="model", y="count",
                text="count",
                color="model",
                title=None,
            )
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)
            opus_share = sli["model_share"].get("claude-opus-4-7", 0)
            haiku_share = sli["model_share"].get("claude-haiku-4-5-20251001", 0)
            st.caption(
                f"Opus share: {opus_share*100:.1f}% (target {SLI['opus_share']['label']})  ·  "
                f"Haiku share: {haiku_share*100:.1f}% (target {SLI['haiku_share']['label']})"
            )
    with cR:
        st.subheader("6 · Cost (estimate)")
        for model, cost in sli["cost_by_model"].items():
            short = model.replace("claude-", "").replace("-20251001", "")
            st.metric(short, f"${cost:.3f}")
        st.caption("Day 7 RouterBench will replace estimates with Langfuse cost API.")

    st.markdown("---")

    # --- Recent runs table ---
    st.subheader("Recent runs")
    summary = (
        df.groupby("run_id")
          .agg(stages=("stage", "nunique"),
               failures=("status", lambda s: (s == "failure").sum()),
               started=("timestamp", "min"),
               finished=("timestamp", "max"))
          .reset_index()
          .sort_values("started", ascending=False)
          .head(20)
    )
    summary["wall_s"] = (summary["finished"] - summary["started"]).dt.total_seconds().round(1)

    # Pull verdict from each run's gate row.
    def _verdict(rid):
        r = df[(df["run_id"] == rid) & (df["stage"] == "gate")]
        if r.empty:
            return "—"
        out = r.iloc[0]["output_json"] or {}
        return out.get("verdict", "—")
    summary["verdict"] = summary["run_id"].apply(_verdict)
    summary = summary[["run_id", "verdict", "stages", "failures", "wall_s", "started"]]
    st.dataframe(summary, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
