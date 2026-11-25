#!/usr/bin/env bash
# Bootstrap the unified Python 3.12 environment at $REPO_ROOT/.venv
# with caches on SSD3 ($HOME/.cache) and editable kayak-gen + kayak-10-9[dev].
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON312_BIN:-/usr/bin/python3.12}"
KAYAKGEN_DIR="${KAYAKGEN_DIR:-$HOME/git/kayak-gen}"

export UV_CACHE_DIR="${UV_CACHE_DIR:-$HOME/.cache/uv}"
export UV_LINK_MODE="${UV_LINK_MODE:-copy}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$HOME/.cache/pip}"
export KAYAK109_CACHE_DIR="${KAYAK109_CACHE_DIR:-$HOME/.cache/kayak-10-9}"

mkdir -p "$HOME/.cache" "$KAYAK109_CACHE_DIR" "$KAYAK109_CACHE_DIR/work"

cache_source="$(findmnt -n -o SOURCE -T "$HOME/.cache" || true)"
ssd3_source="$(findmnt -n -o SOURCE -T /srv/ssd3 || true)"
cache_dev="${cache_source%%\[*}"
ssd3_dev="${ssd3_source%%\[*}"
if [ -z "$cache_dev" ] || [ -z "$ssd3_dev" ] || [ "$cache_dev" != "$ssd3_dev" ]; then
  echo "[setup-env] error: \$HOME/.cache ($cache_source) is not on SSD3 (/srv/ssd3 -> $ssd3_source)" >&2
  exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "[setup-env] error: Python 3.12 not found at $PYTHON_BIN" >&2
  exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup-env] error: uv not found on PATH" >&2
  exit 1
fi

if [ ! -d "$KAYAKGEN_DIR" ]; then
  echo "[setup-env] error: KAYAKGEN_DIR not found at $KAYAKGEN_DIR" >&2
  exit 1
fi

uv venv --python "$PYTHON_BIN" --clear "$REPO_ROOT/.venv"
uv pip install --python "$REPO_ROOT/.venv/bin/python" -e "$KAYAKGEN_DIR" -e "$REPO_ROOT[dev]"

echo "[setup-env] ok: $("$REPO_ROOT/.venv/bin/python" --version) at $REPO_ROOT/.venv (SSD3 cache: $cache_source)"
