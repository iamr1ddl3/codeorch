# Benchmark prompt set

`prompts.jsonl` holds 30 prompts, 10 per complexity tier, half from a public dataset and half adapted from Rocket's public prompt starters. No prompt was written from scratch for this benchmark, so the set isn't tuned to what CodeOrch already does well.

| Tier | WebGen-Bench | Rocket-adapted |
|---|---|---|
| low | 5 | 5 (landing, lead-gen, pitch, vs-page, launch page) |
| moderate | 5 | 5 (onboarding flow, feature prototype, blog, calendar, search) |
| high | 5 | 5 (CRUD invoices, ops dashboard, SaaS MVP, store + cart, signed webhook) |

Rebuild (byte-identical output; pinned dataset revision + fixed seed):

```bash
python benchmark/build_prompt_set.py
```

## Sources

**WebGen-Bench** (Lu et al., 2025 — [paper](https://huggingface.co/papers/2505.03733), [dataset](https://huggingface.co/datasets/luzimu/WebGen-Bench), MIT licence). Test split, revision `c061fe8`. Instructions are used verbatim. Each keeps its `ui_tests` (task + expected result) for the runtime checks that come later. Sampling: browser games excluded (far from Rocket's product surface), then per tier a seeded round-robin across the dataset's three primary categories without repeating an application type.

**Rocket-adapted** (`rocket_adapted.json`). Structure and feature lists follow the Build prompt starters in [Rocket's docs](https://docs.rocket.new/learn/prompt-starters/overview); each record cites its source page. Wording is paraphrased, placeholders are filled with fictional products, and real integrations (Supabase, Stripe, external APIs) are restated as locally mocked because the sandbox has a fixed dependency set.

## Record format

```json
{"id": "R11", "source": "rocket-adapted", "source_ref": "...", "license": "...",
 "tier": "high", "category": "crud", "app_type": "crud",
 "prompt": "Build a full-stack CRUD app for managing invoices ...", "ui_tests": []}
```

## Generation contract (applies identically to every model)

The prompt above is the user message. Every model also gets the same system instructions:

- Next.js 15 App Router, TypeScript (strict), Tailwind CSS 4. Pages under `app/`; shared components under `components/`; path alias `@/*` maps to the project root.
- Only `next`, `react`, `react-dom` and Tailwind are available. No other npm packages; draw charts with SVG and write small utilities by hand.
- No external services or network calls at build time. Databases, auth, payments, email and third-party APIs are mocked with local data, route handlers and in-memory state. Secrets come from `process.env`.
- Output is a JSON object `{"files": {"path": "content", ...}}`; toolchain files (`package.json`, `tsconfig.json`, `next.config.*`, ...) are fixed and ignored if returned.

The output goes straight to `sandbox/runner.py`.

## Running the generation

`generate.py` sends every prompt to every model with the same system contract, saves the raw response, parses the files, build-checks them in the sandbox, and appends one record per sample to `benchmark/runs/<batch>/results.jsonl` (git-ignored).

```bash
# test run: 3 prompts, 1 sample, Claude only, $1.50 cap
python -m benchmark.generate --batch dry --prompts R01,W000028,R11 --models haiku,sonnet --samples 1 --budget 1.5

# full run: 30 prompts x 3 samples x 3 models, $20 hard cap, sandbox in Docker
python -m benchmark.generate --batch b1 --budget 20 --sandbox-mode docker
```

- **Models:** `haiku` = claude-haiku-4-5-20251001 ($1/$5 per MTok), `sonnet` = claude-sonnet-4-6 ($3/$15). `gemini` is enabled only when `GEMINI_API_KEY`, `GEMINI_MODEL`, `GEMINI_PRICE_IN` and `GEMINI_PRICE_OUT` are all set in `.env`, so the budget is never computed on a guessed price.
- **Budget:** before each call the worst case (estimated input + 16k output tokens) is reserved; a call that could exceed `--budget` is not made, even with parallel workers. Re-running a batch resumes and counts earlier spend.
- **Failures are recorded, not dropped:** `parse_error` (no `<file>` blocks), `truncated` (hit the output limit), `provider_error` (retried on resume). Billing/auth errors abort the batch at the first occurrence.
- **Deadline:** Gemini and DeepSeek are streamed with a 20-minute wall-clock limit (`CODEORCH_GEN_DEADLINE_S`); a sample still generating at the limit is recorded as `gen_timeout` and charged its full worst-case reservation.
- **Builds:** at most 2 sandbox builds run at once (`CODEORCH_BUILD_CONCURRENCY`); generation calls run in parallel up to `--workers`.
- **Sampling:** API default temperature (1.0) so the 3 samples per prompt are independent, as pass@k requires.
- **Output format:** `<file path="...">` blocks instead of JSON, so escaping whole source files in JSON isn't a hidden failure mode.

## Known limits

- The WebGen-Bench tier rule (number of UI tests: ≤5 low, 6 moderate, ≥7 high) is a proxy. Some "low" prompts (for example W000003, a multi-company financial dashboard) are harder than the tier suggests. Tiers are for stratification, not a claim about difficulty; per-prompt pass rates will show the real difficulty.
- 30 prompts × 3 samples supports comparisons with wide confidence intervals; it is sized for error analysis, not a leaderboard.
