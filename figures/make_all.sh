#!/usr/bin/env bash
# Regenerate all three figures from the CSV exports in data/.
#
# NOTE: this overwrites figures/out/*.pdf, which are listed in MANIFEST.sha256.
# After re-running, those three manifest entries no longer match; either
# regenerate the manifest (see docs/REPRODUCE.md) or copy the outputs elsewhere
# if you only want to inspect them.
set -euo pipefail
cd "$(dirname "$0")"

python fig1_flow.py
python fig2_accuracy.py
python fig3_swellex.py

echo "figures written to figures/out/"
echo "Reminder: MANIFEST.sha256 now differs for these three PDFs; regenerate it before verifying."
