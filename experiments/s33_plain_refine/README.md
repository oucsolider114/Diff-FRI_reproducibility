# S33 — plain multi-start search as an initialization control

## Question

Within the non-orthogonal (water-column) trial family, does the differentiable joint
refinement of Diff-FRI change the *tracking* stage relative to a plain multi-start grid
search plus a Nelder–Mead local search? A second question follows: greedy grid selection
attains the lowest first-frame error in this configuration, so why is it not used?

## Design

Four initialization arms, all on the same 20 paired realizations of the evolving-SSP
experiment of Sec. S2.1 `(21 frames, 10 dB, 40 receivers, K = 3)`:

| Arm | Description |
|---|---|
| Greedy grid | five-candidate search replaced by the single best coarse candidate |
| Five-candidate | 512-point grid, five candidate sets retained |
| Five-candidate + FRI refinement | the full Diff-FRI initializer |
| Five-candidate + Nelder–Mead | same coarse stage, plain local search on the projection objective, no differentiable joint optimization |

Scripts: `s33_plain_refine.py` (three refinement arms), `s33_greedy_track.py` (greedy arm).
Results: `results/nonorth_init.csv`, `results/greedy_track.csv`.

## Result

Tracking does not separate the arms. Paired median difference between the FRI and
Nelder–Mead arms is 1.35e-9 rad/m (maximum 2.65e-5), the FRI arm is better in 11 of 20
realizations, and the criterion-met fraction is 0.95 for every arm. First-frame error
spans a factor of five across the arms (0.98 to 4.95 times 10^-4 rad/m) while tracking
moves by less than 2%.

Greedy grid attains the lowest first-frame and tracking errors here. It is not used
because it carries no free three-parameter state (hence no residual-floor separation of
Sec. S2.6) and no acceptance rule (hence no noise-scale-free update rule of Sec. S2.3).
Diff-FRI is chosen for those structural properties, not for tracking accuracy.

## Caveat

The 2% tracking difference between the Nelder–Mead and FRI arms is not a systematic
advantage: the paired median is three orders below the first-frame gap and the sign is
not consistent across realizations.
