# CSMT-v2 Independent Holdout

Decision: **PROMOTE_CSMT_V2**

- CSMT-v2 parameter hash: `3dd3f22811c41288c3ec29616b62134e1b9d4d54c055653a9612336cf2189cd5`
- Source frozen CMTR-v1 hash: `e7393cb9cfefbd0b736caf80527fc33e69cee7947a9d7b679fe4541ca94400e1`
- Trials: 20
- New stochastic sequences: YES
- Same Y for all compared front ends: YES
- CSMT-v2 retuned after holdout start: NO
- Reacquire in CSMT-v2: NO
- Reacquire in CMTR-v1 baseline: real Full Proposed call

## Aggregate front end

- Full Proposed kRMSE: 7.968895e-04
- Alpha-beta kRMSE: 7.240172e-04
- CMTR-v1 kRMSE: 2.695370e-04
- CSMT-v2 kRMSE: 2.493917e-04
- Full runtime: 2.8703 s/frame
- CMTR-v1 runtime: 0.2343 s/frame
- CSMT-v2 runtime: 0.1458 s/frame
- CSMT speedup vs Full: 19.684x
- CSMT speedup vs CMTR-v1: 1.607x

## Known-coupling downstream

- Alpha coeff.error: 6.518952e-02
- CMTR-v1 coeff.error: 2.881257e-02
- CSMT-v2 coeff.error: 2.685111e-02
- Alpha correlation: 0.974328
- CMTR-v1 correlation: 0.992123
- CSMT-v2 correlation: 0.994067

## Promotion flags

- k_noninferior_to_cmtr_v1: True
- runtime_better_than_cmtr_v1: True
- no_coupling_noninferior_to_cmtr_v1: True
- known_coupling_noninferior_to_cmtr_v1: True
- correlation_noninferior_to_cmtr_v1: True
- k_supported_vs_full: True
- runtime_supported_vs_full: True
- no_coupling_supported_vs_alpha: True
- known_coupling_supported_vs_alpha: True
- correlation_supported_vs_alpha: True

## Scientific interpretation

PROMOTE_CSMT_V2 means the simplified no-Reacquire architecture met all pre-registered non-inferiority/performance constraints against both frozen CMTR-v1 and the external Full/Alpha-beta baselines.
The result does not erase CMTR-v1; CMTR-v1 remains the development method whose ablation motivated the structural simplification.