#!/usr/bin/env bash
# Run the full unit test suite from the repository root.
set -euo pipefail
cd "$(dirname "$0")/.."
python3 scripts/check_env.py
PYTHONPATH=src python3 -m unittest discover -s tests -v
