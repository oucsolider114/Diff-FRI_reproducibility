# S26 — OCMS vs Diff-FRI initializer comparison

Supplementary Sec. S26 compares a multi-snapshot OCMS-inspired baseline (Li et al., 2026)
with this work's Diff-FRI initializer, on the same 40-m / 300-Hz / 10-dB synthetic frame,
and then tracks the resulting initialization with SMT over the 21-frame evolving-SSP record.

## What this directory contains

| File | Role |
|---|---|
| `s26_rerun.py` | The archived, corrected S26 driver. |
| `ocms_compare_v2.py` | OCMS Algorithms 1–2: mode-depth dictionary by orthogonality + 1-D search over `xi_1`. |
| `ssp_wavenumbers.py` | Reference wavenumbers for a supplied sound-speed profile (KRAKEN-equivalent recurrence). |
| `eps_sweep.py`, `eps_sweep.csv` | Tolerance sensitivity of the OCMS `l1` constraint. |
| `ocms_compare.py`, `ocms_precise.py`, `ocms_v3.py` | Earlier OCMS implementations retained for provenance. |
| `s26_rerun.csv` | Paired per-realization results behind Table S21. |
| `refine_ablation.py`, `refine_ablation_forced.csv` | Paired coarse-vs-refined ablation (gate bypassed) supporting Supplementary Sec. S27. |

## Corrections applied

1. **Estimator identity.** The first version of this comparison called
   `direct_estimator.estimate` (ModeFRINET multistage, 2300 steps, no beam search),
   which is *not* the Diff-FRI described in Sec. 2.3 of the main text.
   `s26_rerun.py` calls `dynamic_topn_refined` — the 512-point coarse grid search
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

## Refinement ablation

The data-fit acceptance test rejected the continuous refinement in all 20 realizations of
this comparison, so the Diff-FRI arm returns its coarse-grid solution. `refine_ablation.py`
isolates the refinement by calling `estimate_from_coarse` directly and tracking both arms
from the same first-frame observation. Across the 20 realizations the refinement raises the
median initialization RMSE from 2.729e-4 to 3.116e-4 rad/m (worse in 17 of 20) while the
median tracking RMSE changes from 4.0642e-4 to 4.0644e-4 rad/m (paired median difference
+6.1e-9, 95% bootstrap interval [-2.8e-9, 1.2e-8]). It costs a median of 22.25 s against
0.63 s for the coarse grid alone.

## Scope check (evolving-SSP negative control)

S26 and S27 use a uniform 1500 m/s column. `precheck_dynamic.py` repeats both comparisons on
the canonical evolving profile (1525/1490 m/s layering, +/-3.7 m thermocline displacement,
reference wavenumbers drifting up to 3.8e-3 rad/m). It writes
`s26_precheck_dynamic.csv` and `s27_precheck_dynamic.csv` and never touches the uniform
results. `eps_sweep_dynamic.py` repeats the OCMS tolerance sweep on that frame.

Two of six checks reverse on the evolving profile: the paired tracking difference no longer
straddles zero (median +8.6e-4 rad/m, 95% interval [9.1e-5, 6.5e-3]), and the forced
refinement changes tracking RMSE by a median 1.5% (max 59%) instead of 0.004%. The other
four keep their direction. The result is recorded in Supplementary Sec. S28 as a scope limit,
not as a robustness confirmation.

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
