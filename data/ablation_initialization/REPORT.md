# CMTR-v1 Component Ablation

- Ablation protocol hash: `d6b0f79a7087acfea07307685e6416e0d0abde982fbbb86191d59fb9a048e727`
- Frozen holdout parameter hash: `e7393cb9cfefbd0b736caf80527fc33e69cee7947a9d7b679fe4541ca94400e1`
- Trials: 20
- Source data: exact completed independent-holdout sequences
- New stochastic observations generated: NO
- Frozen CMTR-v1 retuned: NO
- Future information used: NO
- Explicit bottom acoustic parameters in tested estimators: NO

## Aggregate front-end

- **Full CMTR-v1**: kRMSE=2.845918e-04, success=0.9786, basis MAC=0.997708, runtime=0.2533 s/frame, reacquire=0.0375.
- **w/o Prediction**: kRMSE=2.847690e-04, success=0.9786, basis MAC=0.997703, runtime=0.2720 s/frame, reacquire=0.0375.
- **w/o SSP Awareness**: kRMSE=2.881637e-04, success=0.9786, basis MAC=0.997658, runtime=0.2728 s/frame, reacquire=0.0375.
- **Fixed Local Window**: kRMSE=3.476069e-04, success=0.9643, basis MAC=0.994967, runtime=0.2657 s/frame, reacquire=0.0350.
- **w/o Reacquire Gate**: kRMSE=2.558388e-04, success=0.9857, basis MAC=0.998535, runtime=0.1510 s/frame, reacquire=0.0000.

## Known-coupling downstream

- **Full CMTR-v1**: coeff.error=3.032953e-02, correlation=0.984934, CMR=29.31 dB.
- **w/o Prediction**: coeff.error=3.034536e-02, correlation=0.984932, CMR=29.28 dB.
- **w/o SSP Awareness**: coeff.error=3.062374e-02, correlation=0.984655, CMR=29.21 dB.
- **Fixed Local Window**: coeff.error=3.732707e-02, correlation=0.972234, CMR=28.74 dB.
- **w/o Reacquire Gate**: coeff.error=2.760496e-02, correlation=0.991149, CMR=30.65 dB.

## Component interpretation

- **w/o Prediction**: LIMITED_NON_SIGNIFICANT_TREND (0/4 paired metrics show a significant advantage for full CMTR-v1).
- **w/o SSP Awareness**: COMPONENT_SUPPORTED (4/4 paired metrics show a significant advantage for full CMTR-v1).
- **Fixed Local Window**: PARTIAL_SUPPORT (1/4 paired metrics show a significant advantage for full CMTR-v1).
- **w/o Reacquire Gate**: NO_CLEAR_CONTRIBUTION_UNDER_NOMINAL_PROTOCOL (0/4 paired metrics show a significant advantage for full CMTR-v1).

## Scientific boundary

Ablation variants are deliberately disabled-component versions of the frozen method. They are not retuned competing methods.
A component receiving NO_CLEAR_CONTRIBUTION_UNDER_NOMINAL_PROTOCOL must not be interpreted as universally useless; it only means the completed nominal 10-dB holdout did not isolate a statistically clear contribution for that component.
The independent holdout and operating-envelope results remain the primary evidence for the complete CMTR-v1 method.