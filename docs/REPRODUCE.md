# Reproduction notes

## Environment

Python 3.13 with the pinned versions in `requirements.txt`. No GPU is used; the
differentiable refinement runs on CPU PyTorch. Everything below completes in under
a minute on a laptop.

## Three commands

```bash
pip install -r requirements.txt
python experiments/cadence_audit.py
bash figures/make_all.sh
```

Expected console output:

| Command | Prints |
|---|---|
| `cadence_audit.py` | a four-row table (one per schedule) plus `all four schedules match the archived summary` |
| `fig2_accuracy.py` | `mean ratio = 3.092` to stderr |
| `fig3_swellex.py` | `n=67 csmt=7.26s global=6872.7s ratio=946x` to stderr |

## What to check after running

| Figure | Check |
|---|---|
| Fig. 2(b) | the `mean ratio` line equals 3.09 ± 0.01; no point may sit above the dashed line. The 3.09 is **mean(Full) / mean(CSMT)** = 9.152 / 2.960. The mean of the per-realization ratios is a different number (3.33); the article uses the ratio of means. |
| Fig. 3(a) | the running mean stays within 0.04–0.18 and shows no trend |
| Fig. 3(b) | the two curves are separated by roughly three orders of magnitude |

## Why some numbers cannot be regenerated from `data/`

Three quantities in the article require inputs that are not redistributed here:

| Quantity | Requires |
|---|---|
| raw KRAKEN reference roots, before scoring | the forward model (not redistributed) |
| trial matrices for the invariance check (Supplementary Sec. S12) | the complex arrays (`.npz`, Zenodo) |
| a full re-run of the estimator rather than a recomputation | the pressure arrays (`.npz`, Zenodo) |

The per-frame exports in `data/` already carry the scored quantities, so every number
printed in the article is recomputable from them; what needs the arrays is a re-run.

`data/synthetic/repeat_summary.csv` contains the 20 pairs, so the bootstrap interval *is*
regenerable from the CSV alone; see the optional snippet below.

## Optional: recompute the bootstrap interval

```python
import numpy as np, pandas as pd
rs = pd.read_csv("data/synthetic/repeat_summary.csv")
r = rs.full_overall_rmse_rad_m.values / rs.csmt_overall_rmse_rad_m.values
rng = np.random.default_rng(0)
bs = [rng.choice(r, r.size, replace=True).mean() for _ in range(20000)]
print(np.percentile(bs, [2.5, 97.5]))   # expect approximately [2.59, 3.69]
```

## Regenerating the manifest

```bash
find . -type f -not -path './.git/*' -not -name 'MANIFEST.sha256' \
  -exec sha256sum {} + | sort -k2 > MANIFEST.sha256
sha256sum -c MANIFEST.sha256
```

The manifest is generated, never edited by hand.

## Known limitations

1. `src/mode_fri/` retains the historical experiment modules used during development.
   The three modules exercised by this article are `causal_ssp_modal_tracking.py`,
   `mode_solver.py`, and `mode_init.py`; the rest are kept for provenance.
2. The archived wall-clock timings are comparative within this archive, not
   hardware-independent benchmarks: the original processor model and thread settings
   were not recorded.
3. Raw SWellEx-96 waveforms and SAVEX15 environmental files are not redistributed;
   see the *What is not included* table in `README.md`.
