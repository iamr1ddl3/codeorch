"""Stateless agent context store backed by Postgres + pgvector.

Every CodeOrch agent reads inputs from this store and writes outputs back.
No in-memory state passing — that's the whole architectural point.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from uuid import UUID

import psycopg
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb


@dataclass
class StageRecord:
    id: int
    run_id: UUID
    stage: str
    agent_name: str
    output_json: dict[str, Any] | None
    score: float | None
    status: str
    error: str | None
    timestamp: datetime


class ContextStore:
    def __init__(self, dsn: str | None = None):
        self.dsn = dsn or os.environ["POSTGRES_URL"]

    def _connect(self) -> psycopg.Connection:
        return psycopg.connect(self.dsn, row_factory=dict_row)

    def write_stage(
        self,
        run_id: UUID,
        stage: str,
        agent_name: str,
        output_json: dict[str, Any] | None,
        score: float | None = None,
        status: str = "success",
        error: str | None = None,
    ) -> int:
        """Persist one agent's output. Returns the row id.

        status='success' for normal writes; 'failure' for partial state writes
        from base_agent's exception handler — that's the failure-isolation primitive.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO agent_context
                  (run_id, stage, agent_name, output_json, score, status, error)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    str(run_id),
                    stage,
                    agent_name,
                    Jsonb(output_json) if output_json is not None else None,
                    score,
                    status,
                    error,
                ),
            )
            row = cur.fetchone()
            assert row is not None
            return row["id"]

    def read_stage(self, run_id: UUID, stage: str) -> StageRecord | None:
        """Return the latest record for (run_id, stage), or None if missing.

        A None return is the SLI #2 'cross-agent context loss' signal —
        callers should treat this as a hard error and not silently retry.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_id, stage, agent_name, output_json,
                       score, status, error, timestamp
                FROM agent_context
                WHERE run_id = %s AND stage = %s
                ORDER BY timestamp DESC
                LIMIT 1
                """,
                (str(run_id), stage),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return StageRecord(**row)

    def list_run(self, run_id: UUID) -> list[StageRecord]:
        """All records for a run, ordered by timestamp ascending."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, run_id, stage, agent_name, output_json,
                       score, status, error, timestamp
                FROM agent_context
                WHERE run_id = %s
                ORDER BY timestamp ASC
                """,
                (str(run_id),),
            )
            return [StageRecord(**row) for row in cur.fetchall()]
