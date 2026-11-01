# Build sandbox

Execution-based pass/fail for generated **Next.js + TypeScript** code. This replaces "an LLM read the code and liked it" with "the code type-checks, lints and builds".

## What it checks

Generated files (`{path: content}`, the shape CodeOrch's Coder returns) are overlaid on a pinned template, then three steps run. **All three always run**, so every sample yields all three signals for error analysis.

| Step | Command | Fails on |
|---|---|---|
| `tsc` | `tsc --noEmit` (strict) | type errors, missing modules |
| `eslint` | `eslint .` with `next/core-web-vitals` + `next/typescript` | rule errors (warnings are counted, not failed) |
| `build` | `next build` | bundling and prerender failures |

`next build` is configured to skip its own type-check and lint, so each error is counted once, under the step that owns it.

**Template:** Next.js 15.5.28 (App Router), React 19.1, TypeScript 5.9, Tailwind CSS 4, ESLint 9 — exact versions in `template/package-lock.json`.

## Rules for generated code

- Dependencies are fixed. Generated code **cannot** replace `package.json`, `tsconfig.json`, `next.config.*`, `eslint.config.*`, `postcss.config.*` or `next-env.d.ts`. Attempts are recorded in `rejected_files` (an import of an unlisted package then fails `tsc` and `build`, which is the honest outcome).
- Absolute paths, `..` traversal and writes into `node_modules/`, `.next/` or `.git/` are rejected.
- `app/` at the root is the default layout. If any file is under `src/app/`, the template's root `app/` is removed so the generated one is used.
- An empty generation is `status: "empty_output"`, never a pass.

## Running it

```bash
# one-time: install template deps (needed for local mode and the tests)
cd sandbox/template && npm ci && cd ../..

# build the isolated image (recommended for any real model output)
docker build -t codeorch-sandbox sandbox/

# check a directory of files, or a Coder-output JSON ({"files": {...}})
python -m sandbox.runner path/to/generated_app            # docker if available, else local
python -m sandbox.runner coder_output.json --mode docker
```

From Python:

```python
from sandbox.runner import run_build_check
res = run_build_check({"app/page.tsx": "...", "components/X.tsx": "..."})
res.passed, res.status, res.total_errors, [ (s.step, s.error_count) for s in res.steps ]
```

## Modes and isolation

| Mode | Isolation | Use for |
|---|---|---|
| `docker` | `--network none`, 2 GB RAM, 2 CPUs, 512 pids, non-root user, generated files mounted read-only | **any model output** |
| `local` | none — runs on the host | developing the harness with trusted fixtures |

`next build` executes page modules while prerendering, so untrusted output should only run in `docker` mode. Note that `--network none` also means `next/font/google` cannot download fonts; a generation that depends on it fails the build. That's deliberate: it's a real-world deploy risk, and it's recorded rather than hidden.

## Output

Each run keeps everything under `sandbox/runs/<run_id>/` (git-ignored):

```
result.json      status, per-step exit code / duration / parsed errors, rejected files
out/tsc.log      out/eslint.json   out/eslint.log   out/build.log   out/steps.jsonl
app/  or  gen/   the exact files that were checked
```

These raw logs are the material for failure labelling, so runs are never auto-deleted.

## Tests

```bash
python -m pytest tests/test_sandbox.py -q
```

Unit tests (path safety, parsing, empty output) run anywhere. Two integration tests build `tests/fixtures/sandbox/good_pricing` (must pass all three steps) and `broken_app` (a missing module, a type error, a conditional hook and a `package.json` overwrite — each must be caught by the right step). They skip if the template deps aren't installed.

## Verified

Both fixtures give identical results in `local` and `docker` mode (Docker Desktop on macOS, Apple silicon): `good_pricing` passes all three steps (~9.6 s in Docker); `broken_app` fails with 2 tsc, 1 ESLint and 1 build error, and its `package.json` overwrite is rejected.

## Not yet

- Not wired into the orchestrator yet: the Coder still targets Python. Next step is a Next.js Coder prompt and a `build` stage between Coder and the quality gate, with build errors feeding the retry loop.
- No runtime checks yet (Playwright render, axe-core accessibility, Web Vitals). Those come after the error analysis shows which failure modes matter.
