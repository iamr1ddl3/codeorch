CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS agent_context (
  id          SERIAL PRIMARY KEY,
  run_id      UUID NOT NULL,
  stage       TEXT NOT NULL,
  agent_name  TEXT NOT NULL,
  output_json JSONB,
  score       FLOAT,
  status      TEXT NOT NULL DEFAULT 'success',
  error       TEXT,
  timestamp   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_agent_context_run_stage
  ON agent_context(run_id, stage);

CREATE INDEX IF NOT EXISTS idx_agent_context_run_timestamp
  ON agent_context(run_id, timestamp DESC);
