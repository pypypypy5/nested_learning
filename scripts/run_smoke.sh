#!/usr/bin/env bash
set -euo pipefail

echo "[Smoke] Running pilot_smoke on CPU"
uv run python train.py --config-name pilot_smoke
