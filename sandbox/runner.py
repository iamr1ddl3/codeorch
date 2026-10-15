"""Execution-based build check for generated Next.js + TypeScript code.

Takes the files a model generated ({path: content}, the same shape CodeOrch's
Coder returns), overlays them on a pinned Next.js 15 template, and runs three
checks: `tsc --noEmit`, ESLint (next/core-web-vitals + next/typescript) and
`next build`. Every step runs even if an earlier one fails, so each sample
yields all three signals for error analysis.

Each run keeps its workspace and raw logs under sandbox/runs/<run_id>/ —
those logs are the raw material for failure labelling, so nothing is deleted.

Modes
  docker (default when Docker is available): runs in the codeorch-sandbox image
         with --network none, memory/CPU limits and a non-root user. Use this
         for any untrusted model output — `next build` executes page modules
         while prerendering.
  local: runs on the host against sandbox/template/node_modules. No isolation;
         for developing the harness on trusted fixtures only.

CLI
  python -m sandbox.runner <files_dir | coder_output.json> [--mode local|docker]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SANDBOX_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = SANDBOX_DIR / "template"
RUNS_DIR = SANDBOX_DIR / "runs"
CHECK_SH = SANDBOX_DIR / "check.sh"
IMAGE = os.environ.get("CODEORCH_SANDBOX_IMAGE", "codeorch-sandbox:latest")
TIMEOUT_S = int(os.environ.get("CODEORCH_SANDBOX_TIMEOUT_S", "300"))

# Files the generated code may not replace. Dependencies and toolchain config
# are fixed so every model is judged against the same environment; an attempt
# to change them is recorded (rejected_files) because it is itself a signal.
PROTECTED = {
    "package.json", "package-lock.json", "tsconfig.json", "next-env.d.ts",
    "next.config.js", "next.config.mjs", "next.config.ts",
    "postcss.config.js", "postcss.config.mjs",
    "eslint.config.js", "eslint.config.mjs", ".eslintrc.json",
}
STEPS = ("tsc", "eslint", "build")

_TSC_RE = re.compile(r"^(?P<file>.+?)\((?P<line>\d+),(?P<col>\d+)\): error (?P<code>TS\d+): (?P<msg>.*)$")
# Lines that open a distinct build error. Headers/summaries ("Failed to compile.",
# "Build failed because of webpack errors") are deliberately excluded so one
# root cause counts once.
_BUILD_ERR_RE = re.compile(r"(Module not found|SyntaxError|Type error|Error occurred prerendering|Export encountered|^\s*(\[)?Error:)")


@dataclass
class StepResult:
    step: str
    exit_code: int | None
    duration_ms: int | None
    passed: bool
    error_count: int = 0
    warning_count: int = 0
    errors: list[dict[str, Any]] = field(default_factory=list)  # first 50
    log: str | None = None


@dataclass
class BuildResult:
    run_id: str
    mode: str
    passed: bool
    status: str  # passed | failed | empty_output | timeout | sandbox_error
    steps: list[StepResult]
    total_errors: int
    files_written: list[str]
    rejected_files: list[dict[str, str]]
    layout: str  # "app" | "src"
    wall_ms: int
    run_dir: str
    note: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------- #
# File validation + workspace preparation
# --------------------------------------------------------------------------- #

def _clean_path(raw: str) -> tuple[str | None, str | None]:
    """Normalise a generated path. Returns (path, None) or (None, reason)."""
    p = raw.strip().replace("\\", "/")
    while p.startswith("./"):
        p = p[2:]
    if not p:
        return None, "empty path"
    if p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        return None, "absolute path"
    parts = p.split("/")
    if ".." in parts:
        return None, "path traversal"
    if parts[0] in {"node_modules", ".next", ".git"}:
        return None, f"writes into {parts[0]}/"
    if p in PROTECTED:
        return None, "protected toolchain file"
    return p, None


def validate_files(files: dict[str, str]) -> tuple[dict[str, str], list[dict[str, str]]]:
    ok: dict[str, str] = {}
    rejected: list[dict[str, str]] = []
    for raw, content in files.items():
        path, reason = _clean_path(raw)
        if reason:
            rejected.append({"path": raw, "reason": reason})
        elif not isinstance(content, str):
            rejected.append({"path": raw, "reason": "content is not text"})
        else:
            ok[path] = content
    return ok, rejected


def _write_files(dest: Path, files: dict[str, str]) -> None:
    for rel, content in files.items():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")


def _prepare_local_app(run_dir: Path, files: dict[str, str], layout: str) -> Path:
    app = run_dir / "app"
    shutil.copytree(TEMPLATE_DIR, app, ignore=shutil.ignore_patterns("node_modules", ".next"))
    os.symlink(TEMPLATE_DIR / "node_modules", app / "node_modules")
    if layout == "src":
        shutil.rmtree(app / "app")
    _write_files(app, files)
    return app


# --------------------------------------------------------------------------- #
# Log parsing
# --------------------------------------------------------------------------- #

def _parse_tsc(text: str) -> tuple[int, list[dict[str, Any]]]:
    errs = []
    for line in text.splitlines():
        m = _TSC_RE.match(line.strip())
        if m:
            errs.append({"file": m["file"], "line": int(m["line"]), "col": int(m["col"]),
                         "code": m["code"], "message": m["msg"]})
    return len(errs), errs[:50]


def _parse_eslint(json_path: Path, app_root: str) -> tuple[int, int, list[dict[str, Any]]]:
    if not json_path.exists():
        return 0, 0, []
    try:
        report = json.loads(json_path.read_text() or "[]")
    except json.JSONDecodeError:
        return 0, 0, []
    n_err = n_warn = 0
    errs: list[dict[str, Any]] = []
    for f in report:
        n_err += f.get("errorCount", 0)
        n_warn += f.get("warningCount", 0)
        rel = os.path.relpath(f.get("filePath", ""), app_root) if app_root else f.get("filePath")
        for m in f.get("messages", []):
            if m.get("severity") == 2:
                errs.append({"file": rel, "line": m.get("line"), "col": m.get("column"),
                             "code": m.get("ruleId") or "parse-error", "message": m.get("message")})
    return n_err, n_warn, errs[:50]


def _parse_build(text: str, exit_code: int | None) -> tuple[int, list[dict[str, Any]]]:
    if exit_code == 0:
        return 0, []
    lines = text.splitlines()
    hits = [i for i, l in enumerate(lines) if _BUILD_ERR_RE.search(l)]
    if not hits:  # failed without a recognisable marker: keep the tail
        return 1, [{"code": "build-failed", "message": "\n".join(lines[-15:])}]
    errs = [{"code": "build-error", "message": "\n".join(lines[i:i + 6]).strip()} for i in hits[:10]]
    return len(hits), errs


def _collect(out_dir: Path, app_root: str) -> list[StepResult]:
    recorded: dict[str, dict[str, Any]] = {}
    steps_file = out_dir / "steps.jsonl"
    if steps_file.exists():
        for line in steps_file.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                recorded[rec["step"]] = rec

    def log(name: str) -> str:
        p = out_dir / name
        return p.read_text(errors="replace") if p.exists() else ""

    results = []
    for step in STEPS:
        rec = recorded.get(step)
        code = rec["exit_code"] if rec else None
        dur = rec["duration_ms"] if rec else None
        sr = StepResult(step=step, exit_code=code, duration_ms=dur, passed=(code == 0),
                        log=str(out_dir / (rec["log"] if rec else f"{step}.log")))
        if rec is None:  # step never ran: harness problem, nothing to parse
            results.append(sr)
            continue
        if step == "tsc":
            sr.error_count, sr.errors = _parse_tsc(log("tsc.log"))
        elif step == "eslint":
            sr.error_count, sr.warning_count, sr.errors = _parse_eslint(out_dir / "eslint.json", app_root)
            if code not in (0, None) and sr.error_count == 0:  # crashed (e.g. bad config)
                sr.error_count, sr.errors = 1, [{"code": "eslint-crash", "message": log("eslint.log")[-1500:]}]
        else:
            sr.error_count, sr.errors = _parse_build(log("build.log"), code)
        if code not in (0, None) and sr.error_count == 0:
            sr.error_count = 1
        results.append(sr)
    return results


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (subprocess.SubprocessError, OSError):
        return False


def _run_local(app: Path, out_dir: Path) -> None:
    if not (TEMPLATE_DIR / "node_modules").exists():
        raise RuntimeError("template deps missing: run `npm ci` in sandbox/template first")
    subprocess.run(["bash", str(CHECK_SH), str(app), str(out_dir)],
                   capture_output=True, timeout=TIMEOUT_S, check=False)


def _run_docker(run_dir: Path, files: dict[str, str], out_dir: Path) -> None:
    gen = run_dir / "gen"
    _write_files(gen, files)
    # The container runs as non-root `node`: make inputs readable and the
    # output dir writable regardless of the host's umask / file modes.
    for d, _, fs in os.walk(gen):
        os.chmod(d, 0o755)
        for f in fs:
            os.chmod(os.path.join(d, f), 0o644)
    out_dir.chmod(0o777)
    cmd = [
        "docker", "run", "--rm", "--network", "none",
        "--memory", "2g", "--cpus", "2", "--pids-limit", "512",
        "-v", f"{gen}:/gen:ro", "-v", f"{out_dir}:/out", IMAGE,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S, check=False)
    (run_dir / "docker.log").write_text(
        f"$ {' '.join(cmd)}\nexit={proc.returncode}\n--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}")


def run_build_check(files: dict[str, str], mode: str | None = None,
                    run_id: str | None = None) -> BuildResult:
    """Build-check one generated app. Never raises for model-caused failures."""
    mode = mode or ("docker" if docker_available() else "local")
    run_id = run_id or time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = RUNS_DIR / run_id
    out_dir = run_dir / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()

    valid, rejected = validate_files(files or {})
    layout = "src" if any(p.startswith("src/app/") for p in valid) else "app"

    def finish(status: str, steps: list[StepResult], note: str | None = None) -> BuildResult:
        res = BuildResult(
            run_id=run_id, mode=mode, passed=(status == "passed"), status=status, steps=steps,
            total_errors=sum(s.error_count for s in steps), files_written=sorted(valid),
            rejected_files=rejected, layout=layout,
            wall_ms=int((time.monotonic() - t0) * 1000), run_dir=str(run_dir), note=note,
        )
        (run_dir / "result.json").write_text(json.dumps(res.to_dict(), indent=2))
        return res

    # Honest failure: nothing usable generated is a failure, not a pass.
    if not any(c.strip() for c in valid.values()):
        return finish("empty_output", [], "no non-empty files after validation")

    try:
        if mode == "docker":
            _run_docker(run_dir, valid, out_dir)
            app_root = "/app"
        else:
            app = _prepare_local_app(run_dir, valid, layout)
            _run_local(app, out_dir)
            app_root = str(app)
    except subprocess.TimeoutExpired:
        return finish("timeout", _collect(out_dir, ""), f"exceeded {TIMEOUT_S}s")
    except Exception as exc:  # harness problem, not a model failure
        return finish("sandbox_error", [], f"{type(exc).__name__}: {exc}")

    steps = _collect(out_dir, app_root)
    if all(s.exit_code is None for s in steps):
        dlog = run_dir / "docker.log"
        tail = dlog.read_text(errors="replace")[-800:] if dlog.exists() else ""
        return finish("sandbox_error", steps, "no step results recorded" + (f"\n{tail}" if tail else ""))
    return finish("passed" if all(s.passed for s in steps) else "failed", steps)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def _load_input(src: Path) -> dict[str, str]:
    if src.is_dir():
        return {str(p.relative_to(src)): p.read_text(encoding="utf-8")
                for p in src.rglob("*") if p.is_file()}
    data = json.loads(src.read_text())
    return data.get("files", data)  # Coder output shape or a bare {path: content}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", type=Path, help="directory of generated files, or coder-output JSON")
    ap.add_argument("--mode", choices=["local", "docker"], default=None)
    ap.add_argument("--run-id", default=None)
    args = ap.parse_args(argv)

    res = run_build_check(_load_input(args.source), mode=args.mode, run_id=args.run_id)
    print(f"{res.status.upper()}  run={res.run_id}  mode={res.mode}  errors={res.total_errors}  {res.wall_ms} ms")
    for s in res.steps:
        mark = "ok " if s.passed else "ERR"
        print(f"  [{mark}] {s.step:<6} exit={s.exit_code} errors={s.error_count} warnings={s.warning_count} {s.duration_ms} ms")
        for e in s.errors[:3]:
            loc = f"{e.get('file')}:{e.get('line')}" if e.get("file") else ""
            first = (str(e.get("message") or "").strip().splitlines() or [""])[0]
            print(f"        {e.get('code')} {loc} {first[:110]}")
    for r in res.rejected_files:
        print(f"  [rejected] {r['path']}: {r['reason']}")
    if res.note:
        print(f"  note: {res.note}")
    print(f"  logs: {res.run_dir}")
    return 0 if res.passed else 1


if __name__ == "__main__":
    sys.exit(main())
