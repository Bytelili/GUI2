#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="${ENV_DIR:-$ROOT_DIR/.venv}"
LLAMA_FACTORY_DIR="$ROOT_DIR/LLaMA-Factory"
PYTHON_BIN="${PYTHON_BIN:-}"

python_is_supported() {
  "$1" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)' >/dev/null 2>&1
}

if [[ -x "$ENV_DIR/bin/python" ]] && ! python_is_supported "$ENV_DIR/bin/python"; then
  echo "Removing incompatible environment: $ENV_DIR"
  rm -rf "$ENV_DIR"
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  if [[ -n "$PYTHON_BIN" ]]; then
    PYTHON_CANDIDATES=("$PYTHON_BIN")
  else
    PYTHON_CANDIDATES=(python3.12 python3.11 python3)
  fi

  SELECTED_PYTHON=""
  for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if command -v "$candidate" >/dev/null 2>&1 && python_is_supported "$candidate"; then
      SELECTED_PYTHON="$candidate"
      break
    fi
  done

  if [[ -n "$SELECTED_PYTHON" ]]; then
    echo "Creating virtual environment with $SELECTED_PYTHON"
    "$SELECTED_PYTHON" -m venv "$ENV_DIR"
  elif command -v conda >/dev/null 2>&1; then
    echo "Python >= 3.11 was not found; creating a Python 3.11 Conda environment."
    conda create --yes --prefix "$ENV_DIR" python=3.11 pip
  else
    echo "ERROR: LLaMA-Factory requires Python >= 3.11." >&2
    echo "Install Python 3.11+, or install Conda, then rerun this script." >&2
    exit 1
  fi
fi

PYTHON="$ENV_DIR/bin/python"
if ! python_is_supported "$PYTHON"; then
  echo "ERROR: $PYTHON is not Python >= 3.11." >&2
  exit 1
fi

echo "Using $("$PYTHON" --version) at $PYTHON"
"$PYTHON" -m pip install --upgrade pip wheel setuptools
"$PYTHON" -m pip install -r "$ROOT_DIR/server/requirements-data.txt"
"$PYTHON" -m pip install -e "$LLAMA_FACTORY_DIR"

echo "Server environment ready."
echo "Use this environment with: export PATH=\"$ENV_DIR/bin:\$PATH\""
