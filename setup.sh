#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! ROOT_DIR="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel 2>/dev/null)"; then
  echo "Error: setup.sh must be run inside a Git repository." >&2
  exit 1
fi

cd "$ROOT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d .venv ]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r mmfakebench/requirements.txt

echo "Environment setup complete."
