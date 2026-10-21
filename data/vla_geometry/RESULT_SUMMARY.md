# Controlled VLA geometry sweep v1

Protocol hash: `f11062362dacc92595fefba82c25cec009aa0c08bb56c3db4d29dac42c71ece9`

## Scientific scope

This is a receiver-mask-only audit. The full-array acoustic observation is generated once per trial/frame and then row-masked. No noise is regenerated after masking and no NN-FRI/CSMT parameter is retuned by geometry.

KRAKEN branches are model-reference quantities used only for scoring after estimation. They are not field truth.

## Geometry summary

| Variant | N | Aperture (m) | Static RMSE | Static success | CSMT tracking RMSE | CSMT tracking success |
|---|---:|---:|---:|---:|---:|---:|
| full_40 | 40 | 39.0 | 7.713904e-04 | 0.700 | 2.270006e-04 | 0.995 |
| short_mid_20 | 20 | 19.0 | 1.554528e-02 | 0.000 | 1.269362e-02 | 0.050 |
| short_mid_14 | 14 | 13.0 | 2.812727e-02 | 0.000 | 2.430354e-02 | 0.000 |
| sparse_fullspan_20 | 20 | 39.0 | 7.186786e-04 | 0.750 | 2.950131e-04 | 0.993 |

## Paired differences relative to full array

| Contrast | Metric | Mean difference | 95% bootstrap CI |
|---|---|---:|---:|
| short_mid_20 - full_40 | static_initializer_rmse_rad_m | 1.477389e-02 | [1.118233e-02, 1.833773e-02] |
| short_mid_20 - full_40 | csmt_tracking_only_mean_rmse_rad_m | 1.246662e-02 | [8.309971e-03, 1.660971e-02] |
| short_mid_20 - full_40 | csmt_tracking_only_success_proportion | -9.450000e-01 | [-9.750000e-01, -9.150000e-01] |
| short_mid_14 - full_40 | static_initializer_rmse_rad_m | 2.735588e-02 | [2.691432e-02, 2.781070e-02] |
| short_mid_14 - full_40 | csmt_tracking_only_mean_rmse_rad_m | 2.407654e-02 | [2.378238e-02, 2.439399e-02] |
| short_mid_14 - full_40 | csmt_tracking_only_success_proportion | -9.950000e-01 | [-1.000000e+00, -9.875000e-01] |
| sparse_fullspan_20 - full_40 | static_initializer_rmse_rad_m | -5.271183e-05 | [-1.837311e-04, 5.705371e-05] |
| sparse_fullspan_20 - full_40 | csmt_tracking_only_mean_rmse_rad_m | 6.801257e-05 | [5.693547e-05, 7.917216e-05] |
| sparse_fullspan_20 - full_40 | csmt_tracking_only_success_proportion | -2.500000e-03 | [-7.500000e-03, 0.000000e+00] |

## Figure semantics

- `01_geometry_masks.png`: each point is one retained receiver at its physical depth; columns are the four preregistered masks.
- `02_static_initializer_rmse.png`: each thin connected trajectory is the same stochastic trial across masks; diamonds are across-trial means.
- `03_csmt_tracking_rmse.png`: same paired-trial convention for update-frame CSMT RMSE; diamonds are across-trial means.
- `04_success_proportions.png`: left bar in each pair is first-frame NN-FRI success; right bar is the mean CSMT update-frame success proportion. Both use the same frozen max-component threshold.

## Interpretation rule

Do not tune the algorithm to rescue a weak geometry. A degradation under a shorter contiguous aperture is itself the requested observability/aperture boundary. The sparse-full-span variant helps separate receiver-count loss from physical-aperture loss, but it is still one controlled synthetic environment rather than a universal array-design theorem.
