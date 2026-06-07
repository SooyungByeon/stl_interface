#!/usr/bin/env bash
# Build, solve, and plot a paper mission. Writes output/paper/ex<ID>.png.
# Usage:  ./run_mission.sh <example-id 1|2|3>
set -euo pipefail
cd "$(dirname "$0")"

SEED=4

EX="${1:-1}"
PYTHONPATH=. conda run --no-capture-output -n stl_gnn \
    python -m planner.run_mission --example "$EX" --seed "$SEED"
