#!/usr/bin/env bash
# Launch the Click2STL authoring interface (PyQt6). Needs a display.
set -euo pipefail
cd "$(dirname "$0")"
PYTHONPATH=. conda run --no-capture-output -n stl_gnn python -m interface.app "$@"
