# S26 — OCMS vs Diff-FRI initializer comparison

Supplementary Sec. S26 compares the single-shooting OCMS estimator of Li et al. (2026)
with this work's Diff-FRI initializer, on the same 40-m / 300-Hz / 10-dB synthetic frame,
and then tracks the resulting initialization with CSMT over the 21-frame evolving-SSP record.

## What this directory contains

| File | Role |
|---|---|
| `s26_rerun.py` | The archived, corrected S26 driver. |
| `ocms_compare_v2.py` | OCMS Algorithms 1–2: mode-depth dictionary by orthogonality + 1-D search over `xi_1`. |
| `ssp_wavenumbers.py` | Reference wavenumbers for a supplied sound-speed profile (KRAKEN-equivalent recurrence). |
| `eps_sweep.py`, `eps_sweep.csv` | Tolerance sensitivity of the OCMS `l1` constraint. |
| `ocms_compare.py`, `ocms_precise.py`, `ocms_v3.py` | Earlier OCMS implementations retained for provenance. |

## Corrections applied

1. **Estimator identity.** The first version of this comparison called
   `direct_estimator.estimate` (ModeFRINET multistage, 2300 steps, no beam search),
   which is *not* the Diff-FRI described in Sec. 2.3 of the main text.
   `s26_rerun.py` calls `dynamic_topn_refined` — the 512-point beam search
   (branch 6, width 24, top 5) followed by 100 coefficient-only and 400 joint
   refinement steps — which is the estimator the main text describes.

2. **Snapshot budget.** OCMS is a single-snapshot method: its `l1` program uses one
   pressure vector. Diff-FRI uses the full frame (20 snapshots). To remove the
   asymmetry, OCMS is extended here to all 20 snapshots as a group-sparse program

   ```
   minimize    sum_m || A[m, :] ||_2
   subject to  || Psi(xi_1) A[:, l] - y_l ||_2 <= eps_l ,  l = 1..L
   ```

   with `eps_l` the per-snapshot true noise norm. The retained modal count is still
   taken as the `K` largest row norms (the `|a|` rule), matching the reported procedure.

3. **Realizations.** The first version reported five independent realizations and quoted
   "OCMS initializes 12.5 times more accurately than Diff-FRI" from the single realization
   in which both methods happened to succeed. `s26_rerun.py` runs **20 paired
   coefficient/noise realizations**: both initializers see the same observation in each
   realization, and paired statistics replace the single-run ratio.

4. **Selection of `xi_1` in the multi-snapshot search.** The single-snapshot procedure
   ranks candidate `xi_1` by whether the sparse solution activates exactly `K` atoms.
   That test does not transfer to the group-sparse problem: across the objective basin
   the active-atom count jumps from 2 to 3 while the `l2,1` objective is still decreasing
   toward the true `xi_1`, so filtering on `n_act == K` keeps only the high-objective edge
   of the basin and returns the wrong modal set. The archived driver therefore orders
   candidates by the group-sparse objective alone. On paired realization 0 this changes
   the OCMS initialization error from 2.58e-2 to **4.02e-4 rad/m**.

## Provenance note

The OCMS sources were **not** present in `DiffFRI_reproducibility_v1.1.3.zip` (0 OCMS
entries) and `requirements.txt` did not declare `cvxpy`/`clarabel`. They are included
here and declared in `requirements.txt`. `direct_benchmark.py` and
`configs/canonical_dynamic_benchmark_v1.json`, previously cited by
`src/mode_fri/unified_benchmark.py` and `experiments/canonical_dynamic_benchmark_v1.py`
but absent from the archive, were recovered from `NN_FRI_20261018_Experiments_FullBackup.zip`.

## Running

```bash
python s26_rerun.py --start 0 --end 20 --out s26_rerun.csv
```

Paths at the top of `s26_rerun.py` (`REPO`, `EXTRACT`) assume the archive layout and
must be adjusted to the local checkout.
