#!/usr/bin/env bash
# Regenerate all three figures from the CSV exports in data/.
set -euo pipefail
cd "$(dirname "$0")"

python fig1_flow.py
python fig2_accuracy.py
python fig3_swellex.py

echo "figures written to figures/out/"
