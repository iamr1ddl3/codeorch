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
_FATAL_RE = re.compile(r"credit balance|exceeded your current quota|Insufficient Balance|authentication|"
                       r"invalid x-api-key|permission|not_found_error|API key not valid|"
                       r"\b401\b|\b403\b", re.I)

BUILD_SLOTS = threading.Semaphore(int(os.environ.get("CODEORCH_BUILD_CONCURRENCY", "2")))

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
# Lenient forms seen in real outputs: unclosed blocks, `name=` instead of `path=`.
_OPEN_ANY_RE = re.compile(r'<file\s+(?:path|name)="([^"]+)"\s*>\n?')
# Markdown fence whose first line is a path comment: // app/page.tsx, /* app/globals.css */, {/* ... */}, # ...
_MD_BLOCK_RE = re.compile(r"```[a-zA-Z]*\n(.*?)```", re.S)
_PATH_COMMENT_RE = re.compile(
    r"^\s*(?://|/\*|\{/\*|#|<!--)\s*((?:app|components|lib|src|hooks|types|utils|styles|data|public)/"
    r"[\w./\[\]()@-]+\.(?:tsx?|jsx?|css|json|md|mjs))\s*(?:\*/\}?|-->)?\s*$")


def _strip_fence(body: str) -> str:
    m = _FENCE_RE.match(body)
    return (m.group(1) if m else body).rstrip() + "\n"


_FENCE_LINE_RE = re.compile(r"^\s*```[a-zA-Z]*\s*$")


def _clean_unclosed(body: str) -> str:
    """Body of an unclosed <file> block: drop a leading ```lang line and cut at
    the last bare ``` line (models often close the code fence but not the
    tag, then add prose). Without this the stray fence breaks the file."""
    lines = body.split("\n")
    if lines and _FENCE_LINE_RE.match(lines[0]):
        lines = lines[1:]
    fence_idx = [i for i, l in enumerate(lines) if l.strip() == "```"]
    if fence_idx:
        lines = lines[:fence_idx[-1]]
    return "\n".join(lines).rstrip() + "\n"


def parse_output(text: str, truncated: bool = False) -> tuple[dict[str, str], str, list[str]]:
    """Return (files, format, incomplete_paths).

    format: "strict"  every block is <file path="...">...</file> as instructed
            "unclosed" <file ...> blocks without closing tags / with name=
            "fenced"  markdown code blocks whose first line names the path
            "none"    nothing recognisable (a real failure)
    Lenient forms are accepted because the intent is unambiguous; the format
    is recorded so instruction-following is measured separately from code.
    If the output was truncated, the last block is incomplete and dropped.
    """
    text = text or ""
    strict = {p.strip(): _strip_fence(b) for p, b in _FILE_RE.findall(text)}
    opened = [m.group(1).strip() for m in _OPEN_ANY_RE.finditer(text)]
    if strict and len(strict) == len(opened):
        return strict, "strict", []
    if opened:
        marks = list(_OPEN_ANY_RE.finditer(text))
        files: dict[str, str] = {}
        for i, m in enumerate(marks):
            end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
            body = text[m.end():end].split("</file>")[0]
            files[m.group(1).strip()] = _clean_unclosed(body)
        incomplete = []
        if truncated and "</file>" not in text[marks[-1].end():]:
            incomplete = [marks[-1].group(1).strip()]
            files.pop(incomplete[0], None)
        return files, "unclosed", incomplete
    files = {}
    for body in _MD_BLOCK_RE.findall(text):
        first, _, rest = body.partition("\n")
        m = _PATH_COMMENT_RE.match(first)
        if m:
            files[m.group(1)] = rest.rstrip() + "\n"
    if files:
        incomplete = []
        if truncated and text.count("```") % 2 == 1:  # an unterminated fence at the end
            incomplete = ["<last fenced block>"]
        return files, "fenced", incomplete
    return {}, "none", []


def parse_files(text: str) -> dict[str, str]:
    """Strict-or-lenient file map (see parse_output)."""
    return parse_output(text)[0]


def incomplete_files(text: str, truncated: bool = True) -> list[str]:
    return parse_output(text, truncated)[2]


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
    reasoning_tokens: int = 0  # part of output_tokens spent thinking (0 if not reported)


@dataclass
class ModelSpec:
    key: str
    provider: str
    model_id: str
    price_in: float   # USD per 1M input tokens
    price_out: float  # USD per 1M output tokens (incl. thinking tokens)
    max_out: int = MAX_OUTPUT_TOKENS  # thinking models spend part of this on reasoning

    def cost(self, tin: int, tout: int) -> float:
        return tin / 1e6 * self.price_in + tout / 1e6 * self.price_out


def _anthropic_call(spec: ModelSpec, system: str, user: str, temperature: float) -> Generation:
    from anthropic import Anthropic
    client = Anthropic(max_retries=4)
    t0 = time.monotonic()
    with client.messages.stream(
        # anthropic SDK >=1.x no longer takes `temperature`; the API default
        # (1.0) is what we want for independent pass@k samples.
        model=spec.model_id, max_tokens=spec.max_out,
        system=system, messages=[{"role": "user", "content": user}],
    ) as stream:
        msg = stream.get_final_message()
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return Generation(text, msg.usage.input_tokens, msg.usage.output_tokens,
                      msg.stop_reason or "", int((time.monotonic() - t0) * 1000), msg.model)


GEN_DEADLINE_S = int(os.environ.get("CODEORCH_GEN_DEADLINE_S", "1200"))


class GenerationTimeout(Exception):
    """The model was still generating at the wall-clock deadline."""


def _post_with_retry(url: str, body: dict, headers: dict):
    """Open a streaming POST, retrying 429/5xx before any tokens arrive."""
    import httpx
    client = httpx.Client(timeout=httpx.Timeout(120.0, read=300.0))
    for attempt in range(5):
        req = client.build_request("POST", url, json=body, headers=headers)
        resp = client.send(req, stream=True)
        if resp.status_code not in (429, 500, 502, 503):
            break
        resp.read(); resp.close()
        time.sleep(min(2 ** attempt * 5, 90))
    if resp.status_code >= 400:
        detail = resp.read().decode(errors="replace")[:300]
        resp.close(); client.close()
        raise RuntimeError(f"HTTP {resp.status_code}: {detail}")
    return client, resp


def iter_sse(lines, deadline: float):
    """Yield parsed JSON payloads from `data:` lines; enforce the deadline.
    Streaming matters: DeepSeek keeps a non-streaming connection alive with
    blank lines while thinking, so a read timeout alone never fires."""
    for line in lines:
        if time.monotonic() > deadline:
            raise GenerationTimeout(f"still generating after {GEN_DEADLINE_S}s")
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        yield json.loads(data)


def _gemini_call(spec: ModelSpec, system: str, user: str, temperature: float) -> Generation:
    url = (f"https://generativelanguage.googleapis.com/v1beta/models/{spec.model_id}"
           ":streamGenerateContent?alt=sse")
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": user}]}],
        "generationConfig": {"maxOutputTokens": spec.max_out, "temperature": temperature},
    }
    t0 = time.monotonic()
    client, resp = _post_with_retry(url, body, {"x-goog-api-key": os.environ["GEMINI_API_KEY"]})
    text, usage, finish, model = [], {}, "", spec.model_id
    try:
        for ev in iter_sse(resp.iter_lines(), t0 + GEN_DEADLINE_S):
            cand = (ev.get("candidates") or [{}])[0]
            text += [p.get("text", "") for p in cand.get("content", {}).get("parts", []) if not p.get("thought")]
            finish = cand.get("finishReason") or finish
            usage = ev.get("usageMetadata") or usage  # cumulative; last one wins
            model = ev.get("modelVersion", model)
    finally:
        resp.close(); client.close()
    out = usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)  # thinking billed as output
    return Generation("".join(text), usage.get("promptTokenCount", 0), out, finish,
                      int((time.monotonic() - t0) * 1000), model, usage.get("thoughtsTokenCount", 0))


def _deepseek_call(spec: ModelSpec, system: str, user: str, temperature: float) -> Generation:
    """OpenAI-compatible streaming chat completions. Thinking is DeepSeek's
    default; reasoning arrives as `reasoning_content` deltas (not part of the
    answer) and is billed inside completion_tokens."""
    body = {"model": spec.model_id, "max_tokens": spec.max_out, "temperature": temperature,
            "stream": True, "stream_options": {"include_usage": True},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    t0 = time.monotonic()
    client, resp = _post_with_retry("https://api.deepseek.com/chat/completions", body,
                                    {"Authorization": f"Bearer {os.environ['DEEPSEEK_API_KEY']}"})
    text, usage, finish, model = [], {}, "", spec.model_id
    try:
        for ev in iter_sse(resp.iter_lines(), t0 + GEN_DEADLINE_S):
            model = ev.get("model", model)
            for ch in ev.get("choices") or []:
                text.append((ch.get("delta") or {}).get("content") or "")
                finish = ch.get("finish_reason") or finish
            usage = ev.get("usage") or usage
    finally:
        resp.close(); client.close()
    return Generation("".join(text), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0), finish,
                      int((time.monotonic() - t0) * 1000), model,
                      (usage.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))


PROVIDERS: dict[str, Callable[[ModelSpec, str, str, float], Generation]] = {
    "anthropic": _anthropic_call, "gemini": _gemini_call, "deepseek": _deepseek_call,
}


# Thinking models spend part of the output allowance on reasoning; at 32k,
# 3 of 5 dry-run samples were cut off (one returned no code at all).
THINKING_MAX_OUT = int(os.environ.get("CODEORCH_GEN_THINKING_MAX_TOKENS", "64000"))


def model_specs() -> dict[str, ModelSpec]:
    """Models available given the keys in the environment.

    List prices in USD per 1M tokens, checked 2026-10-15:
      Anthropic  platform.claude.com/docs/en/about-claude/pricing
      Gemini     ai.google.dev/gemini-api/docs/pricing  (3.8 Flash: promo price
                 through 2026-12-31, then $1.50 / $7.50; thinking billed as output)
      DeepSeek   api-docs.deepseek.com/quick_start/pricing  (peak, cache-miss
                 rates; off-peak is half, so recorded costs are upper bounds)
    Override model/price with <KEY>_MODEL / <KEY>_PRICE_IN / <KEY>_PRICE_OUT.
    """
    def env(key: str, model: str, pin: float, pout: float) -> tuple[str, float, float]:
        k = key.upper()
        return (os.environ.get(f"{k}_MODEL", model),
                float(os.environ.get(f"{k}_PRICE_IN", pin)), float(os.environ.get(f"{k}_PRICE_OUT", pout)))

    table = [  # key, provider, required env var, default model, in, out, max_out
        ("haiku", "anthropic", "ANTHROPIC_API_KEY", "claude-haiku-4-5-20251001", 1.00, 5.00, MAX_OUTPUT_TOKENS),
        ("sonnet", "anthropic", "ANTHROPIC_API_KEY", "claude-sonnet-4-6", 3.00, 15.00, MAX_OUTPUT_TOKENS),
        ("gemini", "gemini", "GEMINI_API_KEY", "gemini-3.8-flash", 0.75, 3.75, THINKING_MAX_OUT),
        ("deepseek", "deepseek", "DEEPSEEK_API_KEY", "deepseek-flash", 0.30, 1.20, THINKING_MAX_OUT),
        ("deepseek_pro", "deepseek", "DEEPSEEK_API_KEY", "deepseek-v4-pro", 1.32, 3.96, THINKING_MAX_OUT),
    ]
    specs = {}
    for key, provider, need, model, pin, pout, max_out in table:
        if os.environ.get(need):
            m, i, o = env(key, model, pin, pout)
            specs[key] = ModelSpec(key, provider, m, i, o, max_out)
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
    return spec.cost(est_in, spec.max_out)


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
    except GenerationTimeout as exc:
        # Tokens generated before the cut are probably billed, but the stream
        # never sent usage: charge the full reservation so the cap stays safe.
        budget.settle(reserve, reserve)
        rec.update(status="gen_timeout", error=str(exc), cost_usd=round(reserve, 6),
                   cost_estimated=True, passed=False, build_passed=False)
        return rec
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
    truncated = gen.stop_reason in ("max_tokens", "MAX_TOKENS", "length")
    files, fmt, incomplete = parse_output(gen.text, truncated)
    rec.update(model_id=gen.model_id, input_tokens=gen.input_tokens, output_tokens=gen.output_tokens,
               cost_usd=round(cost, 6), gen_latency_ms=gen.latency_ms, stop_reason=gen.stop_reason,
               reasoning_tokens=gen.reasoning_tokens, truncated=truncated, incomplete_files=incomplete,
               output_format=fmt, format_ok=(fmt == "strict"),
               n_files=len(files), raw=_rel(raw_path))

    if not files:
        rec.update(status="parse_error", build_passed=False, passed=False)
        return rec

    with BUILD_SLOTS:  # builds are CPU/RAM heavy; generation is just waiting
        res = run_build_check(files, mode=sandbox_mode, run_id=f"{batch_dir.name}__{sid}")
    # `passed` is the verdict used in analysis: the build must pass AND the
    # output must be complete. A truncated app that happens to build is
    # missing whatever was cut off, so it is not a pass.
    rec.update(
        status="ok", build_status=res.status, build_passed=res.passed,
        passed=bool(res.passed and not truncated and not incomplete), build_errors=res.total_errors,
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
    """Finished sample ids and total spend. Re-parsed records repeat a
    sample's cost, so spend counts each API call once: provider errors cost 0,
    and for every other sample only its latest record is counted."""
    done: set[str] = set()
    latest_cost: dict[str, float] = {}
    if results.exists():
        for line in results.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if r.get("status") != "provider_error":  # retry provider errors on resume
                    done.add(r["sample_id"])
                    latest_cost[r["sample_id"]] = r.get("cost_usd", 0.0)
    return done, sum(latest_cost.values())


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
        "temperature": temperature, "anthropic_temperature": "API default (1.0)",
        "system_prompt": SYSTEM_PROMPT, "prompt_ids": [p["id"] for p in prompts],
    }, indent=2))

    # Sample-major order: every (prompt, model) gets sample 0 before any gets
    # sample 1. If the budget runs out, the cut removes later samples evenly
    # instead of dropping the last (hardest, high-tier) prompts.
    jobs = [(p, s, k) for k in range(samples) for p in prompts for s in specs
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
            mark = ("PASS" if rec.get("passed") else
                    "TRUNC+BUILT" if rec.get("build_passed") else rec.get("build_status", rec["status"]).upper())
            print(f"[{stats['completed']}/{len(jobs)}] {rec['sample_id']:<28} {mark:<13} "
                  f"${rec.get('cost_usd', 0):.4f}  spent ${budget.spent:.2f}/{budget_cap:.2f}", flush=True)

    stats["spent_usd"] = round(budget.spent, 4)
    return stats


def reparse_batch(batch: str, sandbox_mode: str | None, runs_dir: Path | None = None) -> dict[str, int]:
    """Re-parse saved raw outputs of parse_error samples with the current
    parser and build them. No API calls; appends updated records (analysis
    keeps the last record per sample)."""
    batch_dir = (runs_dir or HERE / "runs") / batch
    results = batch_dir / "results.jsonl"
    last: dict[str, dict[str, Any]] = {}
    for line in results.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            last[r["sample_id"]] = r
    stats = {"reparsed": 0, "recovered": 0}
    for sid, r in last.items():
        if not r.get("raw") or not (r.get("status") == "parse_error"
                                     or r.get("output_format") in ("unclosed", "fenced")):
            continue
        raw = Path(r["raw"]) if Path(r["raw"]).is_absolute() else ROOT / r["raw"]
        files, fmt, incomplete = parse_output(raw.read_text(), bool(r.get("truncated")))
        new = dict(r, output_format=fmt, format_ok=(fmt == "strict"), incomplete_files=incomplete,
                   n_files=len(files), reparsed=True)
        stats["reparsed"] += 1
        if files:
            res = run_build_check(files, mode=sandbox_mode, run_id=f"{batch}__{sid}")
            new.update(status="ok", build_status=res.status, build_passed=res.passed,
                       passed=bool(res.passed and not r.get("truncated") and not incomplete),
                       build_errors=res.total_errors,
                       step_passed={s.step: s.passed for s in res.steps},
                       step_errors={s.step: s.error_count for s in res.steps},
                       rejected_files=[x["path"] for x in res.rejected_files],
                       build_wall_ms=res.wall_ms, sandbox_run=res.run_id)
            stats["recovered"] += 1
        with results.open("a") as f:
            f.write(json.dumps(new) + "\n")
        print(f"reparsed {sid}: format={fmt} files={len(files)} -> "
              f"{'PASS' if new.get('passed') else new.get('build_status', new['status'])}", flush=True)
    return stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Generate + build-check benchmark apps.")
    ap.add_argument("--batch", required=True, help="run name; re-use it to resume")
    ap.add_argument("--models", default="haiku,sonnet,gemini,deepseek")
    ap.add_argument("--prompts", default="", help="comma-separated ids (default: all 30)")
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--budget", type=float, default=20.0, help="hard USD cap for this batch")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--sandbox-mode", choices=["local", "docker"], default=None)
    ap.add_argument("--reparse", action="store_true",
                    help="re-parse + build saved parse_error outputs with the current parser (no API calls)")
    args = ap.parse_args(argv)
    if args.reparse:
        print(json.dumps(reparse_batch(args.batch, args.sandbox_mode)))
        return 0

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
            print(f"skipping model '{key}': its API key is not set")
    if not specs:
        return 1
    stats = run_batch(args.batch, load_prompts([i for i in args.prompts.split(",") if i]), specs,
                      args.samples, args.budget, args.workers, args.temperature, args.sandbox_mode)
    print(json.dumps(stats))
    return 0


if __name__ == "__main__":
    sys.exit(main())
