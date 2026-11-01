<div align="center">


# 🛠️ CodeOrch

### Multi-Agent Code Generation Pipeline

*One natural-language spec in → working, tested, documented code out. Stateless agents over an externalized context store — the property that lets the pipeline survive 3am.*

<br/>

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![Anthropic](https://img.shields.io/badge/Claude-Opus%20%2F%20Sonnet%20%2F%20Haiku-412991.svg?logo=anthropic&logoColor=white)](https://www.anthropic.com/)
[![pgvector](https://img.shields.io/badge/pgvector-HNSW-336791.svg?logo=postgresql&logoColor=white)](https://github.com/pgvector/pgvector)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Langfuse](https://img.shields.io/badge/observability-Langfuse-000000.svg)](https://langfuse.com/)

<br/>

[**Codegen benchmark**](#-codegen-benchmark--in-plain-words) · [**Quick Start**](#-quick-start) · [**Architecture**](#-architecture) · [**8 SLIs**](#-the-8-slis-live-on-the-dashboard) · [**RouterBench**](#-routerbench--ab-benchmark-for-the-quality-gate) · [**Known Limits**](#-scope--known-limits)

</div>

---

## 🧭 Overview

Stateless agents with an externalized context store. One natural-language spec in, working / tested / documented code out — via a planner, parallel coder + tester, an LLM-as-judge quality gate with self-correction, a reviewer, and a documenter. Multi-model routing across Opus 4.7 / Sonnet 4.6 / Haiku 4.5. Operational visibility through an 8-SLI Streamlit dashboard.

Every agent reads inputs from pgvector and writes outputs back. **No in-memory chaining.** A failure mid-pipeline writes a `status='failure'` partial-state record — the next agent sees `read_stage(run_id, ...) is None` and the orchestrator decides retry vs escalate.

---

## 🎓 Codegen benchmark — in plain words

> **Status (2026-11):** DeepSeek and GLM runs complete; Gemini partial; Claude not yet run. Code on branch `feat/build-sandbox`.

**The question:** when you ask an AI model to *build a web app from one sentence* ("a pricing page with 3 tiers and a toggle"), which model gives you an app that actually works — and what does each working app cost?

**Think of it as an exam:**

| Exam part | In this repo |
|---|---|
| **Exam paper** — 30 app requests (10 easy · 10 medium · 10 hard), half from the public [WebGen-Bench](https://huggingface.co/datasets/luzimu/WebGen-Bench) dataset, half modelled on real prompt-to-app product prompts | `benchmark/prompts.jsonl` |
| **Students** — AI models (DeepSeek, GLM, Gemini, Claude); each answers every request 3 times | `benchmark/generate.py` |
| **Marking** — not "does the code look right?" but "does it actually build?": type-check, lint and `next build` in an isolated Docker container | `sandbox/` |
| **Report card** — pass rate (with confidence intervals), cost per *working* app, speed, and *why* answers failed | `benchmark/analyze.py` → `benchmark/runs/<batch>/report.md` |

**Results so far** (batch `b1`, Next.js 15 + TypeScript + Tailwind, 30 prompts × 3 attempts):

| Model | Apps | Built on 1st try (95% CI) | Built within 3 tries | Cost per working app | Main way it fails |
|---|---|---|---|---|---|
| DeepSeek V4.1 Flash | 90 | **59%** (43–74%) | 70% | ≈ $0.05 | runs out of output space on big, open-ended requests |
| GLM-5.3-Flash | 90 | **27%** (13–41%) | 37% | ≈ $0.11 | spends 87% of its output "thinking"; a third of answers contain no code at all |
| Gemini 3.8 Flash | 8 | 75% (38–100%) | — | ≈ $0.12 | too few samples yet |
| Claude Haiku / Sonnet | — | not run yet | | | |

**What it shows so far:**
- **Cheapest per token ≠ cheapest per working app.** GLM-5.3-Flash costs half of DeepSeek per token but about twice as much per app that builds (DeepSeek beats it on the same prompts: 30 wins vs 1, exact McNemar p ≈ 3×10⁻⁸).
- **Clear requests matter more than model choice.** Specific, scoped requests passed far more often than open-ended "build a website for X" ones (DeepSeek 40/45 vs 13/45).
- **Thinking-heavy budget models fail by running out of room, not by writing bad code** — a routing problem you can measure and fix.

**Run it yourself** (macOS: Homebrew Python needs a virtual environment):

```bash
python3 -m venv .venv && source .venv/bin/activate
python -m pip install httpx python-dotenv anthropic
docker build -t codeorch-sandbox sandbox/          # Docker Desktop must be running
python -m benchmark.generate --batch b1 --models deepseek --budget 5 --sandbox-mode docker
python -m benchmark.analyze --batch b1             # → benchmark/runs/b1/report.md
```

API keys go in `.env` (`ANTHROPIC_API_KEY`, `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `GLM_API_KEY`); a model is used only if its key is set. The `--budget` cap is enforced before every call, and re-running the same command resumes where it stopped. Details: [`benchmark/README.md`](benchmark/README.md).

---

## 🧩 Architecture

```mermaid
flowchart TD
    S([spec]) --> P[Planner<br/>Sonnet]
    P --> G{{"asyncio.gather"}}
    G --> C[Coder<br/>Sonnet]
    G --> T[Tester<br/>Sonnet · plan-driven]
    C --> QG[Quality Gate<br/>Haiku · rubric 0–1]
    T --> QG
    QG -->|≥ 0.75| R[Reviewer<br/>Sonnet]
    QG -->|0.50–0.74| RETRY[retry-with-issues<br/>max 2 · Coder only]
    QG -->|< 0.50| ESC[escalate to<br/>Orchestrator]
    RETRY --> C
    R --> D[Documenter<br/>Haiku]
    D --> OUT([documented_code + summary])

    style S fill:#412991,color:#fff
    style OUT fill:#009688,color:#fff
    style ESC fill:#B45309,color:#fff
```

### What's inside

- **Stateless agents over an externalized context store.** Each agent reads its inputs from pgvector and writes its output back. Crash-recovery and post-mortem are queries, not log-grepping.
- **Failure isolation by construction.** `asyncio.gather(coder, tester, return_exceptions=True)` — a Coder crash doesn't take Tester down. Both partial-state rows persist for inspection.
- **Plan-driven Tester (not impl-driven).** Tester reads the spec/plan, not Coder's code, so the gate compares output against an independent reference. Eliminates the circular-validation problem.
- **Quality gate with verdict/routing consistency enforced in code.** The model can produce `score=0.92, verdict="retry"`; we override after parsing so routing stays trustworthy.
- **Self-correction loop.** Score 0.50–0.74 → re-run Coder with the gate's `issues` injected as additional context. Max 2 retries. Tester output reused.
- **Multi-model routing as a first-class concern.** `agents/models.py` is the single source of truth — every agent's model choice has a one-line rationale.
- **Production observability.** Langfuse — one trace per run, sessions, user_id, tags. Token cost auto-computes from `usage_details + model`.

---

## 🚀 Quick Start

```bash
cd codeorch

# 1. boot postgres + pgvector
docker compose up -d

# 2. install deps in a venv
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. populate .env (copy .env.example, fill keys)
cp .env.example .env   # then add ANTHROPIC_API_KEY + LANGFUSE_*

# 4. full-pipeline smoke test (8 stages)
python tests/day6_smoke.py

# 5. boot the API
uvicorn api.main:app --reload --port 8765
curl -X POST localhost:8765/generate -H 'content-type: application/json' \
  -H 'x-user-id: you@example.com' \
  -d '{"spec":"Write a Python function `clamp(x, low, high)`. Include 2 pytest tests."}'
```

<details>
<summary><b>▶ Dashboard + golden benchmark</b></summary>

```bash
# boot the SLI dashboard (separate terminal)
streamlit run dashboard/app.py
# → http://localhost:8501

# run the golden benchmark (requires the API up on port 8765)
npx promptfoo eval --config evals/promptfoo.yaml --output evals/results.json
python evals/promptfoo_runner.py    # → evals/report.html
```

</details>

---

## 📊 The 8 SLIs (live on the dashboard)

| # | SLI | Target | Notes |
|---|---|---|---|
| 1 | Agent failure rate | < 5% | per agent-stage attempt |
| 2 | Cross-agent context loss | < 2% | runs missing any of plan/code/tests/gate |
| 3a | Planner latency (avg) | < 8s | Sonnet 4.6 |
| 3b | Coder latency (avg) | < 30s | Sonnet 4.6 |
| 3c | Gate latency (avg) | < 5s | verdict-only on Haiku 4.5 |
| 4 | Task completion accuracy | ≥ 75% | gate verdict='pass' rate |
| 5 | Hallucination flags | < 10% | gate.issues populated |
| 6 | Cost per task per model | Haiku < $0.01, Sonnet < $0.05, Opus < $0.20 | from Langfuse generation spans |
| 7 | Retry rate | < 20% | runs with retries_used > 0 |
| 8 | Model routing distribution | Opus < 15%, Haiku > 40% | structurally — Haiku owns gate + documenter |

---

## 🏁 RouterBench — A/B benchmark for the quality gate

`routerbench/` ships an A/B harness that replays prior runs through three gate variants — Haiku-weighted-rubric, Haiku-verdict-only, Sonnet-verdict-only — to quantify the cost-quality trade-off in routing decisions.

Headline result on the 13-run benchmark:

| Variant | Mean score | Mean latency | Mean cost | Pass rate |
|---|---|---|---|---|
| Haiku-weighted | 0.903 | 7.4s | $0.00535 | 92% (12/13) |
| **Haiku-verdict-only** | **0.915** | **1.9s** | **$0.00236** | **92% (12/13)** |
| Sonnet-verdict-only | 0.949 | 2.5s | $0.00657 | 92% (12/13)¹ |

<sub>Mean LLM latency and cost per gate call, computed from token counts in `routerbench/results.jsonl` at current list prices (Haiku 4.5 $1/$5, Sonnet 4.6 $3/$15 per MTok). ¹ Sonnet returned prose instead of JSON on 1/13 runs; score, latency and cost are averaged over the 12 parsed runs. At n=13 the score differences are within noise — the latency and cost differences are the robust result.</sub>

Verdict-only on Haiku beat the weighted rubric on speed and cost (3.9× faster, 56% cheaper) with no loss in score. The production gate was switched to verdict-only as a result; `gate_variants.py` keeps the weighted prompt accessible for retry/escalate paths where decomposed scoring aids root-cause.

```bash
python routerbench/run_benchmark.py     # replays 13 runs × 3 variants → results.jsonl
python routerbench/generate_report.py   # renders routerbench_report.html
```

---

## 🧪 Build sandbox — execution-based checks

`sandbox/` checks generated **Next.js + TypeScript** code by running it, not by asking an LLM about it: `tsc --noEmit` → ESLint (`next/core-web-vitals` + `next/typescript`) → `next build`, against a pinned Next.js 15 / React 19 / Tailwind 4 template, in a Docker container with no network. All three steps always run and every log is kept, so each sample gives three independent signals for failure analysis.

```bash
cd sandbox/template && npm ci && cd ../..
docker build -t codeorch-sandbox sandbox/
python -m sandbox.runner path/to/generated_app      # → PASSED / FAILED with per-step errors
python -m pytest tests/test_sandbox.py -q
```

Not yet wired into the orchestrator (the Coder still targets Python). Details and limits: [`sandbox/README.md`](sandbox/README.md).

---

<details>
<summary><b>📁 Repository layout</b></summary>

```
codeorch/
├── docker-compose.yml          postgres 15 + pgvector
├── scripts/init_schema.sql     agent_context table + indexes
├── store/context_store.py      write_stage / read_stage / list_run
├── agents/
│   ├── base_agent.py           sync run() + async arun(); try/catch + partial state writer; shared _generate helper
│   ├── anthropic_client.py     sync + async call wrappers; JSON parse + token usage
│   ├── models.py               central routing table (Opus 4.7 / Sonnet 4.6 / Haiku 4.5)
│   ├── planner.py              Sonnet — spec → structured plan
│   ├── coder.py                Sonnet, async — plan → code files
│   ├── tester.py               Sonnet, async — plan → test files (NOT impl-driven)
│   ├── quality_gate.py         Haiku — verdict-only score + routing
│   ├── gate_variants.py        Haiku-weighted / Haiku-verdict / Sonnet-verdict (used by RouterBench)
│   ├── reviewer.py             Sonnet — final approval + final_code
│   ├── documenter.py           Haiku — adds docstrings; non-blocking on failure
│   └── orchestrator.py         lifecycle: planner → gather(coder,tester) → gate → retry/route → reviewer → documenter
├── observability/langfuse.py   one-trace-per-run; sessions; user_id; tags; cost auto-compute
├── api/main.py                 FastAPI: POST /generate, GET /runs/{id}, /health
├── dashboard/app.py            Streamlit 8-SLI dashboard
├── evals/
│   ├── promptfoo.yaml          10-task golden benchmark (3 easy / 4 medium / 3 hard)
│   └── promptfoo_runner.py     HTML report renderer
├── routerbench/
│   ├── run_benchmark.py        replay-based A/B runner across gate variants
│   ├── generate_report.py      plotly HTML report generator
│   └── routerbench_report.html headline result page
├── sandbox/
│   ├── runner.py               build-check one generated app: tsc → ESLint → next build
│   ├── check.sh                the three steps (same script locally and in Docker)
│   ├── Dockerfile              isolated image: pinned template, deps pre-installed, non-root
│   └── template/               Next.js 15 + TS + Tailwind 4 app the output is overlaid on
├── tests/                      smoke + integration tests (incl. test_sandbox.py + fixtures)
├── requirements.txt
└── .env.example
```

</details>

<details>
<summary><b>🧱 Stack</b></summary>

- **Language:** Python 3.11+
- **Agents:** Anthropic Claude (Opus 4.7 / Sonnet 4.6 / Haiku 4.5) via official SDK; async via `asyncio.gather` for parallel Coder + Tester.
- **Context store:** PostgreSQL + pgvector (HNSW index). Schema documented in `store/`.
- **Observability:** Langfuse (self-hosted or cloud), one trace per pipeline run with per-stage spans.
- **Evals:** Promptfoo (golden dataset, JSON-Schema assertions), in-tree A/B harness for RouterBench.
- **Dashboard:** Streamlit, 8-SLI live view from Langfuse + Postgres.
- **Dev infra:** docker-compose for Postgres + pgvector. Langfuse runs separately (cloud or your own self-hosted instance).

</details>

---

## 🔍 Scope / Known Limits

**In scope.** Stateless multi-agent code generation pipeline over an externalized context store, with quality gating, multi-model routing, observability, and reproducible benchmarks. End-to-end working demo with verifiable SLIs.

**Known limits.**

- Single-language target (Python code generation). Multi-language would need stage-specific Tester adaptations.
- RouterBench n=13 baseline runs — small sample, results directionally robust but worth re-running at n≥50 before treating as definitive.
- No production-grade auth on the dashboard (Streamlit default). Add reverse-proxy auth before exposing publicly.
- Quality gate is LLM-as-judge. Tester **writes** test files but the Python pipeline does not yet **execute** them, so the gate is currently the only pass/fail signal. The new `sandbox/` build checker (tsc → ESLint → `next build`) is the first step toward execution-based verification; see [Build sandbox](#-build-sandbox--execution-based-checks).
- Self-correction is bounded: up to 2 Coder retries (`MAX_RETRIES=2`), stopping early if a retry fails to improve the gate score. Repairs are driven by the gate's `issues`, not by real build or test errors yet.

---

## 📚 Further Reading

- `agents/models.py` — routing table with the per-agent rationale and the RouterBench gate decision.
- `agents/orchestrator.py` — the full lifecycle and retry/escalate logic.
- `sandbox/README.md` — how the execution-based build checker works.

## License

MIT
