#!/usr/bin/env bash
# Regenerate all three figures from the CSV exports in data/.
#
# Output goes to figures/out_repro/, NOT figures/out/. The archived
# figures/out/*.pdf are the evidence covered by MANIFEST.sha256, so this
# script never overwrites them and the manifest keeps verifying afterwards.
set -euo pipefail
cd "$(dirname "$0")"

OUTDIR="out_repro"
mkdir -p "$OUTDIR"

python fig1_flow.py     --stdout > "$OUTDIR/fig1_flow.pdf"
python fig2_accuracy.py --stdout > "$OUTDIR/fig2_accuracy.pdf"
python fig3_swellex.py  --stdout > "$OUTDIR/fig3_swellex.pdf"

echo "figures written to figures/$OUTDIR/"
echo "Archived figures/out/*.pdf are untouched, so MANIFEST.sha256 still verifies."
