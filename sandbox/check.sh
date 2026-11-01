#!/usr/bin/env bash
# Runs the three build checks on a prepared Next.js app and records each one.
#
#   check.sh <app_dir> <out_dir>
#
# Every step runs even if an earlier one fails: for error analysis we want all
# three signals per sample, not just the first failure.
# Writes: <out_dir>/tsc.log, eslint.json, eslint.log, build.log, steps.jsonl
# Timing uses node (always present) so this runs unchanged on macOS and Linux.
set -u
APP="$1"; OUT="$2"
mkdir -p "$OUT"; : > "$OUT/steps.jsonl"
export NEXT_TELEMETRY_DISABLED=1 CI=1
now() { node -e 'process.stdout.write(String(Date.now()))'; }

run_step() {  # name, log file, command...
  local name="$1" log="$2"; shift 2
  local t0 t1 code
  t0=$(now)
  ( cd "$APP" && "$@" ) > "$OUT/$log" 2>&1
  code=$?
  t1=$(now)
  printf '{"step":"%s","exit_code":%d,"duration_ms":%d,"log":"%s"}\n' \
    "$name" "$code" "$((t1 - t0))" "$log" >> "$OUT/steps.jsonl"
}

run_step tsc    tsc.log    npx --no-install tsc --noEmit -p tsconfig.json --pretty false
run_step eslint eslint.log npx --no-install eslint . -f json -o "$OUT/eslint.json"
run_step build  build.log  npx --no-install next build
exit 0
