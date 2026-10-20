# Data dictionary

All CSV files are UTF-8 with a BOM (`utf-8-sig`) and use `snake_case` columns.
Units are rad/m for wavenumbers and seconds for times unless stated otherwise.

---

## `data/synthetic/frame_metrics.csv`

One row per (realization, frame). Produced by the evolving-waveguide experiment of Sec. 3.1.

| Column | Meaning |
|---|---|
| `repeat` | realization index, 0–19 |
| `time_index` | frame index, 0–20 |
| `time_s` | elapsed time in seconds (60 s per frame) |
| `kraken_k1..k3` | reference modal horizontal wavenumbers |
| `full_k1..k3` | framewise independent estimation |
| `csmt_k1..k3` | sequential estimate |
| `full_k_rmse_rad_m`, `csmt_k_rmse_rad_m` | per-frame RMSE over the three components |
| `full_success`, `csmt_success` | 1 if every component is within 1.5e-3 rad/m |
| `csmt_predicted_k1..k3` | prediction passed to the local search |
| `csmt_ssp_change` | relative SSP change driving the window width |
| `csmt_mean_search_half_width_rad_m` | δ used at that frame |
| `local_runtime_seconds` | wall-clock time of the update |

## `data/synthetic/repeat_summary.csv`

One row per realization; the source of the 3.09 factor and of Fig. 2(b).

| Column | Meaning |
|---|---|
| `repeat` | realization index |
| `full_overall_rmse_rad_m` | all-frame RMSE, framewise estimation |
| `csmt_overall_rmse_rad_m` | all-frame RMSE, CSMT |
| `csmt_track_only_rmse_rad_m` | RMSE excluding the initialization frame |
| `full_frame_success_rate`, `csmt_frame_success_rate`, `csmt_track_only_success_rate` | success fractions |
| `csmt_track_runtime_mean_s`, `csmt_track_runtime_p95_s` | update timing |

## `data/robustness/sweeps_summary.csv`

The 12 conditions of the update-stage sweep (Supplementary Secs. S16, S23): five SNRs, four
depth shifts, and three extra snapshot counts, with the nominal condition counted once.

| Column | Meaning |
|---|---|
| `condition` | row index |
| `factor` | `snr`, `snapshots`, `shift`, `nominal` |
| `snr_db`, `snapshots`, `shift_m` | factor levels |
| `rmse` | pooled RMSE over the 400 update frames |
| `ci_low`, `ci_high` | paired bootstrap interval |
| `success` | fraction of updates within the criterion |
| `outside` | fraction of reference values outside the pre-correction window |

## `data/robustness/ssp_perturbation_summary.csv`

The separate offset / thermocline-contrast audit (Supplementary Sec. S25).

| Column | Meaning |
|---|---|
| `condition` | row index |
| `perturbation` | `nominal`, `offset`, or `contrast` |
| `delta_m_s` | applied Δc |
| `rmse_pooled` | pooled RMSE over 20 trials × 20 frames × 3 components |
| `bootstrap_mean_lo`, `bootstrap_mean_hi` | interval on the repeat mean |
| `success` | fraction of the 400 frames meeting the criterion |
| `ratio_to_original_nominal` | ratio to the unperturbed condition |

`ssp_perturbation_frame_metrics.csv` holds the per-frame records behind these two tables;
`ssp_perturbation_protocol.json` is the protocol fingerprint.

## `data/swellex/csmt_fixed_ssp_i9606.csv`

67 uninterrupted updates of SWellEx-96 Event S5 under a fixed i9606 profile.
First row is `state = INIT_FULL`; the rest are `state = TRACK`.

| Column | Meaning |
|---|---|
| `original_frame` | SWellEx frame index |
| `state` | `INIT_FULL` or `TRACK` |
| `k1_rad_m..k3_rad_m` | accepted estimates |
| `pred1_rad_m..pred3_rad_m` | predictions |
| `width1_rad_m..width3_rad_m` | search half-widths |
| `ssp_change` | relative SSP change |
| `residual_mean`, `residual_worst` | normalized projection residual |
| `local_runtime_seconds`, `total_runtime_seconds` | timing |

## `data/swellex/Table_final_experimental_summary.csv`

`Metric,Value` pairs. The two medians used in Fig. 3(b) are
`CSMT runtime median (s)` and `Representative NNFR runtime median (s)`.

## `data/swellex/cadence_frame_metrics.csv`

Per-frame export of the fixed-cadence audit (Supplementary Sec. S24). 4 schedules × 68 frames.

| Column | Meaning |
|---|---|
| `schedule` | `no_reset`, `period20`, `period10`, `period05` |
| `frame` | SWellEx frame index, 2–69 |
| `fixed_mae` | MAE under the fixed initial mode-order assignment |
| `unique_mae` | MAE under many-to-one nearest-root matching |
| `retained` | 1 if all three assignments are retained at that frame |
| `residual` | normalized projection residual |
| `runtime` | wall-clock time of that frame (includes re-initialization when it occurs) |
| `k1..k3` | accepted wavenumbers |

Frame 2 carries the mandatory initialization and is excluded from the scored set
(67 frames).

## `data/swellex/cadence_summary.csv`

The archived summary of the same audit; columns `fixed_mae`, `unique_mae`,
`retained`, `scored_frames`, `residual`, `runtime`. Used as the regression target
by `experiments/cadence_audit.py`.

## `data/swellex/cadence_audit.csv`

Produced by `experiments/cadence_audit.py` and checked against `cadence_summary.csv`.

| Column | Meaning |
|---|---|
| `schedule` | `no_reset`, `period20`, `period10`, `period05` |
| `scored_frames` | number of scored frames (67) |
| `fixed_mae` | mean of `fixed_mae` over scored frames |
| `unique_mae` | mean of `unique_mae` |
| `retained` | number of frames retaining all three assignments |
| `residual` | mean normalized residual |
| `runtime` | summed wall-clock time of all frames |

## `data/baselines/estimator_comparison.csv`

Table I. One row per estimator class, RMSE in 1e-3 rad/m.
