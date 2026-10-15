"""Generate apps for every (prompt, model, sample) and build-check each one.

    python -m benchmark.generate --batch b1                      # full run
    python -m benchmark.generate --batch dry --prompts R01,W000015 --models haiku --samples 1

For each sample it saves the raw model response, parses the files, runs
sandbox/runner.py on them and appends one record to
benchmark/runs/<batch>/results.jsonl. Re-running the same batch resumes:
finished (prompt, model, sample) triples are skipped and their spend still
counts toward the budget.

Budget: before every call the worst case (input + max output tokens at list
price) is reserved; a call that could push spend past --budget is not made.
So the cap holds even with parallel workers.

Output format: models return files as <file path="...">...</file> blocks
rather than JSON — escaping whole source files inside JSON is a failure mode
of its own and would blur the comparison. A response with no parseable files
is recorded as parse_error (an honest failure), not skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from sandbox.runner import run_build_check  # noqa: E402

# Errors that will hit every remaining call too (billing, auth, bad model id):
# stop the batch instead of recording hundreds of identical failures.
_FATAL_RE = re.compile(r"credit balance|authentication|invalid x-api-key|permission|not_found_error|API key not valid|"
                       r"\b401\b|\b403\b", re.I)

MAX_OUTPUT_TOKENS = int(os.environ.get("CODEORCH_GEN_MAX_TOKENS", "16000"))

SYSTEM_PROMPT = """You are generating a complete, working web application.

Stack (fixed — you cannot change it):
- Next.js 15 App Router, React 19, TypeScript with "strict": true, Tailwind CSS 4 (already imported in app/globals.css).
- Available packages: next, react, react-dom, tailwindcss. Nothing else is installed. Do not import any other package; write small utilities yourself and draw charts with SVG.
- Path alias "@/*" maps to the project root.
- A default app/layout.tsx that imports "./globals.css" exists; you may replace it.
- package.json, tsconfig.json, next.config.*, eslint and postcss configs are fixed and ignored if you return them.

Rules:
- Pages go under app/, shared components under components/, helpers under lib/.
- No external services or network calls. Mock databases, auth, payments, email and third-party APIs with local data, Next.js route handlers and in-memory state. Read secrets from process.env.
- Do not use next/font/google (no network at build time).
- The app must pass `tsc --noEmit`, ESLint (next/core-web-vitals + next/typescript) and `next build`.

Output format — return every file in full, each in its own block, and nothing else:
<file path="app/page.tsx">
...file content...
</file>
"""

_FILE_RE = re.compile(r'<file\s+path="([^"]+)"\s*>\n?(.*?)\n?</file>', re.S)
_FENCE_RE = re.compile(r"^\s*```[a-zA-Z]*\n(.*)\n```\s*$", re.S)


def parse_files(text: str) -> dict[str, str]:
    files: dict[str, str] = {}
    for path, body in _FILE_RE.findall(text or ""):
        m = _FENCE_RE.match(body)  # tolerate a code fence inside the block
        files[path.strip()] = (m.group(1) if m else body).rstrip() + "\n"
    return files


# --------------------------------------------------------------------------- #
# Providers
# --------------------------------------------------------------------------- #

@dataclass
class Generation:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str
    latency_ms: int
    model_id: str


@dataclass
class ModelSpec:
    key: str
    provider: str
    model_id: str
    price_in: float   # USD per 1M input tokens
    price_out: float  # USD per 1M output tokens (incl. thinking tokens)

    def cost(self, tin: int, tout: int) -> float:
        return tin / 1e6 * self.price_in + tout / 1e6 * self.price_out


def _anthropic_call(spec: ModelSpec, system: str, user: str, temperature: float) -> Generation:
    from anthropic import Anthropic
    client = Anthropic(max_retries=4)
    t0 = time.monotonic()
    with client.messages.stream(
        # anthropic SDK >=1.x no longer takes `temperature`; the API default
        # (1.0) is what we want for independent pass@k samples.
        model=spec.model_id, max_tokens=MAX_OUTPUT_TOKENS,
        system=system, messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return Generation(text, msg.usage.input_tokens, msg.usage.output_tokens,
                      msg.stop_reason or "", int((time.monotonic() - t0) * 1000), msg.model)


def _gemini_call(spec: ModelSpec, system: str, user: str, temperature: float) -> Generation:
    import httpx
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{spec.model_id}:generateContent"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": MAX_OUTPUT_TOKENS, "temperature": temperature},
    }
    t0 = time.monotonic()
    for attempt in range(5):
        r = httpx.post(url, json=body, timeout=600,
                       headers={"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
        if r.status_code not in (429, 500, 503):
            break
        time.sleep(2 ** attempt * 3)
    r.raise_for_status()
    d = r.json()
    cand = (d.get("candidates") or [{}])[0]
    text = "".join(p.get("text", "") for p in cand.get("content", {}).get("parts", []))
    u = d.get("usageMetadata", {})
    out_tokens = u.get("candidatesTokenCount", 0) + u.get("thoughtsTokenCount", 0)  # thinking is billed as output
    return Generation(text, u.get("promptTokenCount", 0), out_tokens, cand.get("finishReason", ""),
                      int((time.monotonic() - t0) * 1000), d.get("modelVersion", spec.model_id))


PROVIDERS: dict[str, Callable[[ModelSpec, str, str, float], Generation]] = {
    "anthropic": _anthropic_call, "gemini": _gemini_call,
}


def model_specs() -> dict[str, ModelSpec]:
    """List prices from platform.claude.com/docs/en/about-claude/pricing (2026-10-15).
    Gemini has no default: set GEMINI_MODEL, GEMINI_PRICE_IN and GEMINI_PRICE_OUT
    from Google's current price page, so the budget cap is never computed on a guess."""
    specs = {
        "haiku": ModelSpec("haiku", "anthropic", "claude-haiku-4-5-20251001", 1.00, 5.00),
        "sonnet": ModelSpec("sonnet", "anthropic", "claude-sonnet-4-6", 3.00, 15.00),
    }
    if all(os.environ.get(k) for k in ("GEMINI_API_KEY", "GEMINI_MODEL", "GEMINI_PRICE_IN", "GEMINI_PRICE_OUT")):
        specs["gemini"] = ModelSpec("gemini", "gemini", os.environ["GEMINI_MODEL"],
                                    float(os.environ["GEMINI_PRICE_IN"]), float(os.environ["GEMINI_PRICE_OUT"]))
    return specs


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #

class Budget:
    """Reserve worst-case cost before a call; settle to the actual cost after."""

    def __init__(self, cap: float, spent: float = 0.0):
        self.cap, self.spent, self.reserved = cap, spent, 0.0
        self._lock = threading.Lock()

    def reserve(self, amount: float) -> bool:
        with self._lock:
            if self.spent + self.reserved + amount > self.cap:
                return False
            self.reserved += amount
            return True

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self.reserved -= reserved
            self.spent += actual


def worst_case_cost(spec: ModelSpec, system: str, user: str) -> float:
    est_in = (len(system) + len(user)) // 3 + 50  # chars/3 over-estimates tokens for English
    return spec.cost(est_in, MAX_OUTPUT_TOKENS)


# --------------------------------------------------------------------------- #
# One sample
# --------------------------------------------------------------------------- #

def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def run_one(prompt: dict[str, Any], spec: ModelSpec, sample: int, batch_dir: Path,
            budget: Budget, temperature: float, sandbox_mode: str | None,
            call: Callable[[ModelSpec, str, str, float], Generation] | None = None) -> dict[str, Any] | None:
    call = call or PROVIDERS[spec.provider]
    sid = f"{prompt['id']}__{spec.key}__s{sample}"
    reserve = worst_case_cost(spec, SYSTEM_PROMPT, prompt["prompt"])
    if not budget.reserve(reserve):
        return None

    rec: dict[str, Any] = {
        "sample_id": sid, "prompt_id": prompt["id"], "model": spec.key, "sample": sample,
        "tier": prompt["tier"], "source": prompt["source"], "category": prompt["category"],
        "temperature": temperature, "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    try:
        gen = call(spec, SYSTEM_PROMPT, prompt["prompt"], temperature)
    except Exception as exc:  # provider error after retries: record, don't charge
        budget.settle(reserve, 0.0)
        msg = f"{type(exc).__name__}: {exc}"
        rec.update(status="provider_error", error=msg[:500], cost_usd=0.0, fatal=bool(_FATAL_RE.search(msg)))
        return rec

    cost = spec.cost(gen.input_tokens, gen.output_tokens)
    budget.settle(reserve, cost)
    raw_path = batch_dir / "raw" / f"{sid}.txt"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text(gen.text)
    files = parse_files(gen.text)
    truncated = gen.stop_reason in ("max_tokens", "MAX_TOKENS")
    rec.update(model_id=gen.model_id, input_tokens=gen.input_tokens, output_tokens=gen.output_tokens,
               cost_usd=round(cost, 6), gen_latency_ms=gen.latency_ms, stop_reason=gen.stop_reason,
               truncated=truncated, n_files=len(files), raw=_rel(raw_path))

    if not files:
        rec.update(status="parse_error", build_passed=False)
        return rec

    res = run_build_check(files, mode=sandbox_mode, run_id=f"{batch_dir.name}__{sid}")
    rec.update(
        status="ok", build_status=res.status, build_passed=res.passed, build_errors=res.total_errors,
        step_passed={s.step: s.passed for s in res.steps},
        step_errors={s.step: s.error_count for s in res.steps},
        rejected_files=[r["path"] for r in res.rejected_files],
        build_wall_ms=res.wall_ms, sandbox_run=res.run_id,
    )
    return rec


# --------------------------------------------------------------------------- #
# Batch
# --------------------------------------------------------------------------- #

def load_prompts(ids: list[str] | None) -> list[dict[str, Any]]:
    prompts = [json.loads(l) for l in (HERE / "prompts.jsonl").read_text().splitlines() if l.strip()]
    if ids:
        want = set(ids)
        prompts = [p for p in prompts if p["id"] in want]
        missing = want - {p["id"] for p in prompts}
        if missing:
            raise SystemExit(f"unknown prompt ids: {sorted(missing)}")
    return prompts


def load_done(results: Path) -> tuple[set[str], float]:
    done, spent = set(), 0.0
    if results.exists():
        for line in results.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                spent += r.get("cost_usd", 0.0)
                if r.get("status") != "provider_error":  # retry provider errors on resume
                    done.add(r["sample_id"])
    return done, spent


def run_batch(batch: str, prompts: list[dict[str, Any]], specs: list[ModelSpec], samples: int,
              budget_cap: float, workers: int, temperature: float, sandbox_mode: str | None,
              call: Callable | None = None, runs_dir: Path | None = None) -> dict[str, Any]:
    batch_dir = (runs_dir or HERE / "runs") / batch
    batch_dir.mkdir(parents=True, exist_ok=True)
    results = batch_dir / "results.jsonl"
    done, spent = load_done(results)
    budget = Budget(budget_cap, spent)
    (batch_dir / "config.json").write_text(json.dumps({
        "models": {s.key: s.__dict__ for s in specs}, "samples": samples, "budget_usd": budget_cap,
        "temperature": temperature, "max_output_tokens": MAX_OUTPUT_TOKENS,
        "system_prompt": SYSTEM_PROMPT, "prompt_ids": [p["id"] for p in prompts],
    }, indent=2))

    jobs = [(p, s, k) for p in prompts for s in specs for k in range(samples)
            if f"{p['id']}__{s.key}__s{k}" not in done]
    write_lock = threading.Lock()
    stop = threading.Event()
    stats = {"completed": 0, "skipped_budget": 0, "already_done": len(done), "aborted": None}

    def work(job):
        if stop.is_set():
            return "stopped"
        p, s, k = job
        rec = run_one(p, s, k, batch_dir, budget, temperature, sandbox_mode, call)
        if isinstance(rec, dict) and rec.get("fatal"):
            stop.set()  # set in the worker so no other job starts after this one
        return rec

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for fut in as_completed([pool.submit(work, j) for j in jobs]):
            rec = fut.result()
            if rec == "stopped":
                continue
            if rec is None:
                stats["skipped_budget"] += 1
                continue
            with write_lock, results.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            stats["completed"] += 1
            if rec.get("fatal") and not stats["aborted"]:
                stats["aborted"] = rec["error"][:300]
                print(f"ABORTING batch — non-retryable provider error: {rec['error'][:200]}", flush=True)
            mark = "PASS" if rec.get("build_passed") else rec.get("build_status", rec["status"]).upper()
            print(f"[{stats['completed']}/{len(jobs)}] {rec['sample_id']:<28} {mark:<13} "
                  f"${rec.get('cost_usd', 0):.4f}  spent ${budget.spent:.2f}/{budget_cap:.2f}", flush=True)

    stats["spent_usd"] = round(budget.spent, 4)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate + build-check benchmark apps.")
    ap.add_argument("--batch", required=True, help="run name; re-use it to resume")
    ap.add_argument("--models", default="haiku,sonnet,gemini")
    ap.add_argument("--prompts", default="", help="comma-separated ids (default: all 30)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--budget", type=float, default=20.0, help="hard USD cap for this batch")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sandbox-mode", choices=["local", "docker"], default=None)
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass
    available = model_specs()
    specs = []
    for key in [m.strip() for m in args.models.split(",") if m.strip()]:
        if key in available:
            specs.append(available[key])
        else:
            print(f"skipping model '{key}': not configured (see model_specs docstring)")
    if not specs:
        return 1
    stats = run_batch(args.batch, load_prompts([i for i in args.prompts.split(",") if i]), specs,
                      args.samples, args.budget, args.workers, args.temperature, args.sandbox_mode)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
