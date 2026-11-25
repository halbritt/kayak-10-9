#!/usr/bin/env bash
# Print the Python interpreter the tests and lint tasks use.
# Order: KAYAK109_PY, this worktree's .venv, the primary checkout's .venv.
# A linked worktree has no .venv of its own, so the last step lets mise
# tasks run there without setting KAYAK109_PY by hand.
set -euo pipefail

if [ -n "${KAYAK109_PY:-}" ]; then
  printf '%s\n' "$KAYAK109_PY"
  exit 0
fi
root="$(git rev-parse --show-toplevel)"
if [ -x "$root/.venv/bin/python" ]; then
  printf '%s\n' "$root/.venv/bin/python"
  exit 0
fi
common="$(git rev-parse --path-format=absolute --git-common-dir)"
primary="$(dirname "$common")"
if [ -x "$primary/.venv/bin/python" ]; then
  printf '%s\n' "$primary/.venv/bin/python"
  exit 0
fi
echo "[test-python] error: no interpreter; create .venv (scripts/setup-env.sh) or set KAYAK109_PY" >&2
exit 1
