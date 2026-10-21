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

## Reproduction tiers

Not every reported result is reachable at the same level. The three tiers are:

| Tier | What runs | Which results |
|---|---|---|
| **A. Table and figure recomputation** | Python from the archived CSV only | Fig. 1–3, Table I, the S23/S25 sweep tables, the S24 cadence audit, the bootstrap interval |
| **B. Estimator re-run on archived pressures** | needs `data/arrays/Y_frames.npz` **and** the original `SWellEx96_S5_CSMT/` tree; not self-contained | the cadence audit via `experiments/rerun_cadence_from_arrays.py` — **fails in a bare checkout** |
| **C. Cold-start experiment re-run** | needs the original project layout, the KRAKEN binary, and the raw external inputs; **not** self-contained in this archive | Sec. S10 StageA/StageB, Sec. S20 geometry sweep |

Not covered at Tier A: the K=3/5/8 retained-mode sweep of Sec. S2. Its per-trial records were
not exported; only the three pooled RMSE values are quoted in the supplement.

Tier A is verified in a clean checkout. Tier B has been exercised on the development machine
but depends on modules that remain in `src/mode_fri/` and on the original `SWellEx96_S5_CSMT/`
tree (`processed/Y_frames.npz`, `results/49hz_kraken_reference_stage2/*.env`); it does not run
from a bare checkout. Tier C is documented by the archived
arrays and per-realization CSVs, which are the authoritative outputs; the scripts under
`data/savex15/scripts/` and `data/vla_geometry/scripts/` are retained for provenance and expect
the original project layout (`--project-root`, `--stagea-root`, and the cold-start caches).
They are **not** guaranteed to run standalone. If you need Tier C, treat the archived NPZ/CSV
as the result and the scripts as documentation of how it was produced.

## Why some numbers cannot be regenerated from `data/`

Three quantities in the article require inputs that are not redistributed here:

| Quantity | Requires |
|---|---|
| raw KRAKEN reference roots, before scoring | the forward model (not redistributed) |
| trial matrices for the invariance check (Supplementary Sec. S12) | the complex arrays (regenerable from `data/arrays/Y_frames.npz`) |
| a full re-run of the estimator rather than a recomputation | `data/arrays/Y_frames.npz` (included, 456 KB) |

The per-frame exports in `data/` already carry the scored quantities, so every number
printed in the article is recomputable from them. The pressure array in `data/arrays/` is
included as well, but `experiments/rerun_cadence_from_arrays.py` also needs the original
`SWellEx96_S5_CSMT/` tree and `dynamic_local_basis_coupling_suite`; it does not run from a
bare checkout, so treat it as Tier C provenance rather than a working entry point.

`data/synthetic/repeat_summary.csv` contains the 20 pairs, so the bootstrap interval *is*
regenerable from the CSV alone; see the optional snippet below.

## Optional: recompute the bootstrap interval

```python
import numpy as np, pandas as pd
rs = pd.read_csv("data/synthetic/repeat_summary.csv")
fv = rs.full_overall_rmse_rad_m.values
cv = rs.csmt_overall_rmse_rad_m.values
rng = np.random.default_rng(0)
bs = np.empty(20000)
for i in range(20000):
    idx = rng.integers(0, fv.size, fv.size)   # same indices for both methods
    bs[i] = fv[idx].mean() / cv[idx].mean()   # ratio of means, not mean of ratios
print(np.percentile(bs, [2.5, 97.5]))         # [2.59, 3.67]
```

The resampling unit is the paired realization, and both methods must be resampled with the
same indices. Averaging the per-realization ratios instead gives 3.33 with an interval near
[2.83, 3.86]; that is a different statistic and is not what the article reports.

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
