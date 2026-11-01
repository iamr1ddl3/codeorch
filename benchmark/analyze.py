"""Summarise a generation batch: pass rates with CIs, cost per passing app,
failure categories and the most common error codes.

    python -m benchmark.analyze --batch b1          # writes runs/b1/report.md + summary.json

Statistics
  pass@1      mean of `passed` over all samples of a model
  pass@k      unbiased estimator (Chen et al., 2021) per prompt, averaged:
              1 - C(n-c, k) / C(n, k) with n samples, c passing
  95% CI      cluster bootstrap over prompts (resample prompts, keep each
              prompt's samples together), 5000 draws, fixed seed
  McNemar     exact two-sided test on (prompt, sample) pairs both models ran

These are automatic categories from the build logs. They are the starting
point for human error analysis (open coding of the raw outputs), not a
replacement for it.
"""

from __future__ import annotations

import argparse
import json
import re
import random
from collections import Counter, defaultdict
from math import comb
from pathlib import Path
from statistics import median
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SANDBOX_RUNS = ROOT / "sandbox" / "runs"
TIERS = ("low", "moderate", "high")


def load(batch: str) -> list[dict[str, Any]]:
    """Last record per sample_id; provider errors only if nothing better exists."""
    best: dict[str, dict[str, Any]] = {}
    for line in (HERE / "runs" / batch / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        prev = best.get(r["sample_id"])
        if prev is None or prev.get("status") == "provider_error" or r.get("status") != "provider_error":
            best[r["sample_id"]] = r
    return list(best.values())


def category(r: dict[str, Any]) -> str:
    """One primary failure category per sample, in causal order."""
    if r.get("passed"):
        return "pass"
    st = r.get("status")
    if st in ("provider_error", "gen_timeout", "parse_error"):
        return st
    if r.get("truncated") or r.get("incomplete_files"):
        return "truncated"
    if r.get("build_status") in ("sandbox_error", "timeout", "empty_output"):
        return f"sandbox:{r.get('build_status')}"
    sp = r.get("step_passed") or {}
    if not sp.get("tsc", True):
        return "type_error"
    if not sp.get("build", True):
        return "build_error"
    if not sp.get("eslint", True):
        return "lint_only"
    return "other"


def pass_at_k(n: int, c: int, k: int) -> float:
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def bootstrap_ci(by_prompt: dict[str, list[bool]], stat, draws: int = 5000, seed: int = 7) -> tuple[float, float]:
    keys = sorted(by_prompt)
    if not keys:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    vals = sorted(stat([by_prompt[rng.choice(keys)] for _ in keys]) for _ in range(draws))
    return vals[int(0.025 * draws)], vals[int(0.975 * draws) - 1]


def mean_pass(groups: list[list[bool]]) -> float:
    flat = [x for g in groups for x in g]
    return sum(flat) / len(flat) if flat else float("nan")


def mean_pass_at(k: int):
    def f(groups: list[list[bool]]) -> float:
        vals = [pass_at_k(len(g), sum(g), k) for g in groups if len(g) >= k]
        return sum(vals) / len(vals) if vals else float("nan")
    return f


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value from discordant counts b, c."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def error_codes(r: dict[str, Any]) -> list[tuple[str, str]]:
    run = r.get("sandbox_run")
    p = SANDBOX_RUNS / run / "result.json" if run else None
    if not p or not p.exists():
        return []
    res = json.loads(p.read_text())
    out = []
    for s in res.get("steps", []):
        seen = set()
        for e in s.get("errors", []):
            code = e.get("code") or "?"
            if code == "build-error":  # first line identifies the kind
                msg = re.sub(r"\x1b\[[0-9;]*m", "", e.get("message") or "")
                code = re.sub(r"^Error:\s*x?\s*", "", msg.splitlines()[0] if msg else "")[:60]
            if code not in seen:  # count each code once per sample
                seen.add(code)
                out.append((s["step"], code))
    return out


def _ensure_format(r: dict[str, Any]) -> None:
    """Older records lack output_format: classify from the saved raw output."""
    if "output_format" in r or not r.get("raw"):
        return
    from benchmark.generate import parse_output
    raw = Path(r["raw"]) if Path(r["raw"]).is_absolute() else ROOT / r["raw"]
    if raw.exists():
        r["output_format"] = parse_output(raw.read_text(), bool(r.get("truncated")))[1]
        r["format_ok"] = r["output_format"] == "strict"


def summarise(recs: list[dict[str, Any]]) -> dict[str, Any]:
    for r in recs:
        _ensure_format(r)
    models = sorted({r["model"] for r in recs})
    summary: dict[str, Any] = {"models": {}, "pairs": {}}
    for m in models:
        rs = [r for r in recs if r["model"] == m and r.get("status") != "provider_error"]
        by_prompt: dict[str, list[bool]] = defaultdict(list)
        for r in rs:
            by_prompt[r["prompt_id"]].append(bool(r.get("passed")))
        n_pass = sum(r.get("passed", False) for r in rs)
        cost = sum(r.get("cost_usd", 0.0) for r in rs)
        tiers = {}
        for t in TIERS:
            tr = [bool(r.get("passed")) for r in rs if r["tier"] == t]
            tiers[t] = {"n": len(tr), "pass_rate": sum(tr) / len(tr) if tr else None}
        cats = Counter(category(r) for r in rs)
        codes = Counter(f"{s}:{c}" for r in rs if not r.get("passed") for s, c in error_codes(r))
        gen = [r["gen_latency_ms"] / 1000 for r in rs if r.get("gen_latency_ms")]
        summary["models"][m] = {
            "n_samples": len(rs), "n_prompts": len(by_prompt), "n_passed": n_pass,
            "pass_at_1": mean_pass(list(by_prompt.values())),
            "pass_at_1_ci95": bootstrap_ci(by_prompt, mean_pass),
            "pass_at_3": mean_pass_at(3)(list(by_prompt.values())),
            "pass_at_3_ci95": bootstrap_ci(by_prompt, mean_pass_at(3)),
            "cost_usd": round(cost, 4),
            "cost_per_passing_app": round(cost / n_pass, 4) if n_pass else None,
            "truncation_rate": sum(bool(r.get("truncated")) for r in rs) / len(rs) if rs else None,
            "median_gen_s": median(gen) if gen else None,
            "median_output_tokens": median([r["output_tokens"] for r in rs if r.get("output_tokens")]) if rs else None,
            "median_reasoning_share": median([r["reasoning_tokens"] / r["output_tokens"] for r in rs
                                              if r.get("output_tokens") and r.get("reasoning_tokens") is not None]) if rs else None,
            "format_ok_rate": (sum(bool(r.get("format_ok")) for r in rs if "output_format" in r)
                               / max(1, sum(1 for r in rs if "output_format" in r))),
            "formats": dict(Counter(r.get("output_format", "pre-v2") for r in rs)),
            "by_tier": tiers, "categories": dict(cats), "top_error_codes": codes.most_common(15),
            "provider_errors": sum(1 for r in recs if r["model"] == m and r.get("status") == "provider_error"),
        }
    idx = {(r["model"], r["prompt_id"], r["sample"]): bool(r.get("passed"))
           for r in recs if r.get("status") != "provider_error"}
    for i, a in enumerate(models):
        for b in models[i + 1:]:
            keys = [(p, s) for (m, p, s) in idx if m == a and (b, p, s) in idx]
            only_a = sum(idx[(a, p, s)] and not idx[(b, p, s)] for p, s in keys)
            only_b = sum(idx[(b, p, s)] and not idx[(a, p, s)] for p, s in keys)
            summary["pairs"][f"{a}_vs_{b}"] = {"paired": len(keys), "only_a_passes": only_a,
                                               "only_b_passes": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b)}
    return summary


def pct(x: float | None) -> str:
    return "—" if x is None or x != x else f"{100 * x:.0f}%"


def render(batch: str, s: dict[str, Any]) -> str:
    lines = [f"# Batch `{batch}` — automatic summary", "",
             "Automatic categories from build logs; the starting point for human error analysis.", "",
             "| Model | Samples | pass@1 (95% CI) | pass@3 (95% CI) | Cost | Cost / passing app | Truncated | Median gen time | Median reasoning share |",
             "|---|---|---|---|---|---|---|---|---|"]
    for m, d in s["models"].items():
        lo, hi = d["pass_at_1_ci95"]
        lo3, hi3 = d["pass_at_3_ci95"]
        cpp = f"${d['cost_per_passing_app']:.3f}" if d["cost_per_passing_app"] else "—"
        lines.append(f"| {m} | {d['n_samples']} | {pct(d['pass_at_1'])} ({pct(lo)}–{pct(hi)}) | "
                     f"{pct(d['pass_at_3'])} ({pct(lo3)}–{pct(hi3)}) | ${d['cost_usd']:.2f} | {cpp} | "
                     f"{pct(d['truncation_rate'])} | {d['median_gen_s']:.0f}s | {pct(d['median_reasoning_share'])} |")
    lines += ["", "Recorded costs are upper bounds where a provider bills off-peak at a discount (DeepSeek).", "",
              "## Pass rate by tier", "", "| Model | low | moderate | high |", "|---|---|---|---|"]
    for m, d in s["models"].items():
        t = d["by_tier"]
        lines.append(f"| {m} | " + " | ".join(f"{pct(t[x]['pass_rate'])} (n={t[x]['n']})" for x in TIERS) + " |")
    lines += ["", "## Output format compliance", "",
              "Instructed format is `<file path=...>...</file>`. Lenient forms are still parsed and built; "
              "this measures instruction-following separately from code quality.", ""]
    for m, d in s["models"].items():
        lines.append(f"**{m}:** strict {pct(d['format_ok_rate'])} — " +
                     ", ".join(f"{k} {v}" for k, v in sorted(d["formats"].items(), key=lambda kv: -kv[1])))
        lines.append("")
    lines += ["## Failure categories (one per sample)", ""]
    for m, d in s["models"].items():
        lines.append(f"**{m}:** " + ", ".join(f"{k} {v}" for k, v in sorted(d["categories"].items(), key=lambda kv: -kv[1])))
        lines.append("")
    lines += ["## Most common error codes in failing samples (counted once per sample)", ""]
    for m, d in s["models"].items():
        lines.append(f"**{m}:**")
        lines += [f"- `{c}` × {n}" for c, n in d["top_error_codes"]] or ["- none"]
        lines.append("")
    if s["pairs"]:
        lines += ["## Paired comparison (McNemar, exact)", "", "| Pair | Paired samples | Only A passes | Only B passes | p |",
                  "|---|---|---|---|---|"]
        for k, v in s["pairs"].items():
            lines.append(f"| {k} | {v['paired']} | {v['only_a_passes']} | {v['only_b_passes']} | {v['mcnemar_p']:.3f} |")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", required=True)
    args = ap.parse_args(argv)
    s = summarise(load(args.batch))
    out = HERE / "runs" / args.batch
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    (out / "report.md").write_text(render(args.batch, s))
    print(render(args.batch, s))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
