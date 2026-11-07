# S30 — number of modewise sweeps

## Question

The main text fixes the local update at **two** modewise sweeps. Does one
sweep already suffice, and is a third redundant?

## Design

Environment and seeds follow the canonical dynamic benchmark
(`configs/dynamic_local_basis_coupling_suite.json`): 40 m water column,
300 Hz, 40 receivers on 1 m, 21 frames, 20 realisations, tracking SNR 30 dB.

The first-frame state is read from `data/synthetic/frame_metrics.csv`
(`csmt_k1..3` at `time_index = 0`), so that all sweep counts start from the
identical state on identical observations.  Rebuilding the SSP trajectory
from the config reproduces the archived thermocline displacement exactly
(max absolute difference 0.0), and the reconstructed first-frame RMSE is
$6.784\times10^{-4}$ rad/m, matching the main text.

Two generating grids are used:

| condition | generating grid | tracking grid |
|---|---|---|
| `native`  | 1.0 m  | 1.0 m |
| `fine025` | 0.25 m | 1.0 m |

## Results

Update-frame RMSE (frames 1–20), mean over 20 realisations, in $10^{-4}$ rad/m.

| condition | seeds | 1 sweep | 2 sweeps | 3 sweeps | 1→2 | 2→3 |
|---|---|---|---|---|---|---|
| native  | A | 1.0700 | 0.6682 | 0.6561 | −37.6 % | −1.8 % |
| native  | B | 0.9970 | 0.6810 | 0.6767 | −31.7 % | −0.6 % |
| fine025 | A | 1.0678 | 0.6685 | 0.6579 | −37.4 % | −1.6 % |
| fine025 | B | 0.9956 | 0.6824 | 0.6780 | −31.5 % | −0.6 % |

One sweep is clearly insufficient; the second sweep recovers 31.5–37.6 % of
the update-frame RMSE; the third changes it by 0.6–1.8 % in every cell.

A convergence check extends the sweep count to 5 and 10. Those rows are in
`results/s30_sweep_summary.csv` but are not tabulated in the supplement:

| condition | seeds | 5 sweeps | 10 sweeps | 3 → 5 |
|---|---|---|---|---|
| native  | A | 0.6558 | 0.6558 | −0.05 % |
| native  | B | 0.6767 | 0.6767 | 0.00 % |
| fine025 | A | 0.6576 | 0.6576 | −0.05 % |
| fine025 | B | 0.6780 | 0.6780 | 0.00 % |

Three sweeps are at the plateau (within 0.05 % of the 10-sweep value) and two
are within 2 % of it. The two sweeps used in the main text are therefore a
cost–accuracy choice, not the converged optimum.

## Limitations

- **Self-consistent setting.** The observations are synthesized with the same
  water-column recurrence that the tracker uses, perturbed only by additive
  complex noise. This is a controlled sensitivity check, not a forward-model
  test.
- **Absolute level.** Update-frame RMSE here is $6.6$–$6.8\times10^{-5}$ rad/m,
  and the all-frame RMSE at two sweeps is $2.07\times10^{-4}$, against
  $2.960\times10^{-4}$ in the main text. The difference arises because the
  generating and fitting models coincide here. Only **relative** differences
  between sweep counts are interpreted, and none of these numbers is combined
  with the main results.
- **The two solvers are identical at fixed $k$.** `solve_mode` and
  `solve_mode_impedance` return pointwise identical columns (maximum absolute
  difference $0$) because they differ only in how the root is selected. No
  basis-mismatch condition can be constructed from them, which is why this
  script does not attempt one.
- **Grid refinement is not a mismatch.** Refining the generating grid from
  1.0 m to 0.25 m changes the result by at most 0.2 %, because the cosine
  finite-difference recurrence is exact for a constant sound-speed interval
  at any step. The `fine025` row is a step-size control, not a forward-model
  mismatch.
- **No KRAKEN forward mismatch.** The full pipeline compares KRAKEN fields
  against a water-column basis. That mismatch is absent here and is not
  reproduced by this script.
- **First-frame state.** `k0` is the archived first-frame estimate, not a
  re-run of the Diff-FRI initialiser, so initialisation cost is excluded.
- **Seed sets.** Two disjoint seed schedules (A and B) are used; they are not
  the schedules of the main benchmark.

## Reproduce

```bash
python experiments/s30_sweep_ablation/s30_sweep_ablation.py
```

The script depends only on `numpy` and `repo/src/mode_fri`; it does not need
`torch` or `matplotlib`. Output: `results/s30_sweep_summary.csv`.
