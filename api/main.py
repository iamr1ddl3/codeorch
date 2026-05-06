"""CodeOrch FastAPI surface.

Day 3 ships the routes and the run-id lifecycle. Day 4 wires the actual
Orchestrator agent into POST /generate.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=True)

from agents.base_agent import AgentFailure  # noqa: E402
from agents.orchestrator import Orchestrator  # noqa: E402
from observability.langfuse import trace_run  # noqa: E402
from store.context_store import ContextStore  # noqa: E402

app = FastAPI(title="CodeOrch", version="0.1.0")
store = ContextStore()


class GenerateRequest(BaseModel):
    spec: str = Field(..., min_length=1, description="Natural language code spec.")


class GenerateResponse(BaseModel):
    run_id: UUID
    status: str
    created_at: datetime
    gate_verdict: str | None = None
    gate_score: float | None = None
    approved: bool | None = None
    documented_code: str | None = None
    doc_summary: str | None = None


class StageView(BaseModel):
    stage: str
    agent_name: str
    status: str
    score: float | None
    output_json: dict | None
    error: str | None
    timestamp: datetime


class RunView(BaseModel):
    run_id: UUID
    stages: list[StageView]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse, status_code=202)
def generate(
    req: GenerateRequest,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> GenerateResponse:
    """Kick off a new run.

    Day 4: dispatches to the Orchestrator (Opus 4) which invokes the Planner
    (Sonnet 4) and writes both stages to pgvector. Future days add the
    Coder/Tester/Gate/Reviewer/Documenter pipeline behind the orchestrator's
    routing decision.
    """
    run_id = uuid4()
    user_id = x_user_id or "anon"

    orch_output: dict | None = None
    final_status = "accepted"

    with trace_run(run_id=run_id, spec=req.spec, user_id=user_id) as root:
        store.write_stage(
            run_id=run_id,
            stage="spec",
            agent_name="api",
            output_json={"spec": req.spec},
            status="success",
        )
        try:
            orch_output = Orchestrator(store).run(run_id, {"spec": req.spec})
            final_status = orch_output.get("status", "accepted")
            root.update(
                output={
                    "run_id": str(run_id),
                    "gate_verdict": orch_output.get("gate_verdict"),
                    "gate_score": orch_output.get("gate_score"),
                    "approved": orch_output.get("approved"),
                    "retries_used": orch_output.get("retries_used"),
                    "code_files": orch_output.get("code_files"),
                    "test_files": orch_output.get("test_files"),
                },
                metadata={"day": 6, "phase": "full-pipeline"},
            )
        except AgentFailure:
            final_status = "failed"
            root.update(
                output={"run_id": str(run_id), "status": "failed"},
                level="ERROR",
            )

    o = orch_output or {}
    return GenerateResponse(
        run_id=run_id,
        status=final_status,
        created_at=datetime.utcnow(),
        gate_verdict=o.get("gate_verdict"),
        gate_score=o.get("gate_score"),
        approved=o.get("approved"),
        documented_code=o.get("documented_code"),
        doc_summary=o.get("doc_summary"),
    )


@app.get("/runs/{run_id}", response_model=RunView)
def get_run(run_id: UUID) -> RunView:
    records = store.list_run(run_id)
    if not records:
        raise HTTPException(status_code=404, detail="run_id not found")
    return RunView(
        run_id=run_id,
        stages=[
            StageView(
                stage=r.stage,
                agent_name=r.agent_name,
                status=r.status,
                score=r.score,
                output_json=r.output_json,
                error=r.error,
                timestamp=r.timestamp,
            )
            for r in records
        ],
    )
