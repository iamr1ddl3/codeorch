"""Build benchmark/prompts.jsonl — the 30-prompt CodeOrch benchmark.

  15 prompts  WebGen-Bench test split (MIT), used verbatim, pinned revision,
              stratified + seeded sample. Their UI test cases are kept for
              later runtime checks.
  15 prompts  adapted from Rocket's public Build prompt starters
              (rocket_adapted.json).

Tiers (Rocket's credit-complexity language: low / moderate / high):
  WebGen-Bench  by number of UI test cases: <=5 low, 6 moderate, >=7 high
                (a proxy for how many distinct behaviours the app must have)
  Rocket        assigned per prompt in rocket_adapted.json

Re-running with the same SEED and REVISION reproduces the file exactly.
    python benchmark/build_prompt_set.py
"""

from __future__ import annotations

import io
import json
import random
import urllib.request
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
REVISION = "c061fe8326459846ac020ceb54d750c859cb059f"  # luzimu/WebGen-Bench main, 2025-09-29
URL = f"https://huggingface.co/datasets/luzimu/WebGen-Bench/resolve/{REVISION}/data/test-00000-of-00001.parquet"
SEED = 20261015
PER_TIER = 5
# Browser games are far from Rocket's product surface (sites and business apps).
EXCLUDE_TYPES = {"Browser-Based Games"}


def tier_for(n_ui_tests: int) -> str:
    return "low" if n_ui_tests <= 5 else "moderate" if n_ui_tests == 6 else "high"


def load_webgen() -> list[dict]:
    with urllib.request.urlopen(URL, timeout=60) as r:
        return pq.read_table(io.BytesIO(r.read())).to_pylist()


def sample_webgen(rows: list[dict]) -> list[dict]:
    """Per tier: round-robin over primary categories, never repeating an
    application type within a tier, seeded for reproducibility."""
    rng = random.Random(SEED)
    pools: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: r["id"]):
        if r["application_type"] in EXCLUDE_TYPES:
            continue
        pools[tier_for(len(r["ui_instruct"]))][r["Category"]["primary_category"]].append(r)

    picked = []
    for tier in ("low", "moderate", "high"):
        cats = {c: rng.sample(v, len(v)) for c, v in sorted(pools[tier].items())}
        seen_types: set[str] = set()
        chosen: list[dict] = []
        while len(chosen) < PER_TIER and any(cats.values()):
            for c in sorted(cats):
                while cats[c] and cats[c][0]["application_type"] in seen_types:
                    cats[c].pop(0)
                if cats[c] and len(chosen) < PER_TIER:
                    r = cats[c].pop(0)
                    seen_types.add(r["application_type"])
                    chosen.append(r)
        for r in chosen:
            picked.append({
                "id": f"W{r['id']}",
                "source": "webgen-bench",
                "source_ref": f"luzimu/WebGen-Bench@{REVISION[:7]} test id {r['id']}",
                "license": "MIT",
                "tier": tier,
                "category": r["Category"]["primary_category"],
                "app_type": r["application_type"],
                "prompt": r["instruction"],
                "ui_tests": [{"task": u["task"], "expected": u["expected_result"]} for u in r["ui_instruct"]],
            })
    return picked


def load_rocket() -> list[dict]:
    data = json.loads((HERE / "rocket_adapted.json").read_text())
    return [{
        "id": p["id"], "source": "rocket-adapted",
        "source_ref": f"docs.rocket.new/{p['source_ref']}", "license": "adapted (paraphrased)",
        "tier": p["tier"], "category": p["category"], "app_type": p["category"],
        "prompt": p["prompt"], "ui_tests": [],
    } for p in data["prompts"]]


def main() -> None:
    prompts = sample_webgen(load_webgen()) + load_rocket()
    order = {"low": 0, "moderate": 1, "high": 2}
    prompts.sort(key=lambda p: (order[p["tier"]], p["source"], p["id"]))
    out = HERE / "prompts.jsonl"
    out.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts))
    counts = defaultdict(int)
    for p in prompts:
        counts[(p["tier"], p["source"])] += 1
    print(f"wrote {len(prompts)} prompts -> {out}")
    for k in sorted(counts, key=lambda k: (order[k[0]], k[1])):
        print(f"  {k[0]:<9} {k[1]:<15} {counts[k]}")


if __name__ == "__main__":
    main()
