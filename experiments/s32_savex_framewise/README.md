# S32 — SAVEX15 framewise control (40 m and 101 m)

## Question

Does the framewise margin reported in the main text survive a change of
environment? S10 already supplies a second (101 m) and third (40 m SAVEX15)
configuration, but only against the held initial estimate.

## Design

- Observations rebuilt exactly as in the Stage-B archive: same SHA256 seed
  schedule (`NN_FRI_Bv1_{C,N}`), 20 snapshots, 10 dB, KRAKEN receiver mode
  shapes.
- 21 frames are sampled every seventh frame and 5 of the 20 realisations are
  used: 35 paired frames per geometry.
- Both arms share the identical observation: the framewise arm runs the full
  initializer (beam 512, branch 6, beam 24, top 5, then the acceptance-tested
  refinement); the SMT arm is read from the archived `CSMT_dynamic_SSP` rows.
- `homogeneous_scale` and the geometry transfer of the grid separation follow
  the Stage-B protocol.

## Results (35 paired frames per geometry)

| geometry | framewise mean | SMT mean | ratio | SMT lower | criterion-met fw / SMT |
|---|---|---|---|---|---|
| design_40 (8 modes) | 9.23e-4 | 8.67e-4 | 1.06 | 22/35 | 0.63 / 0.74 |
| sav15_101 (19 modes) | 2.15e-3 | 1.09e-3 | 1.97 | 25/35 | 0.20 / 0.54 |

## What it shows

1. The 3.09 factor of the main text is **not universal**: on the 40 m SAVEX15
   configuration the two arms are comparable (1.06x, 22/35).
2. At 101 m the margin returns (1.97x, 25/35) and the criterion-met fraction
   separates them (0.54 vs 0.20). Here framewise estimation of a 19-mode field
   with three retained modes is ill-conditioned: the per-frame estimate is
   2.15e-3, worse than simply holding the frame-0 estimate (1.04e-3).
3. The advantage therefore tracks how ill-conditioned single-frame estimation
   is, not a property of the update rule alone.

## Limitations

- 35 paired frames per geometry (7 frames x 5 realisations), not the full
  144 x 20; chosen because one full estimate costs about 51 s on these
  profiles.
- Observations are KRAKEN-synthesised, so the forward model matches the
  generator better than in the main-text synthetic experiment. This is the
  likely reason the 40 m margin shrinks.
- The SMT arm is archived output, not re-run inside this script; the pairing
  is exact because the observations are regenerated with the archived seeds.

## Reproduce

```bash
S32_OUT=./results python s32_savex_framewise.py --geom design_40 --reps 0 5 --stride 21
S32_OUT=./results python s32_savex_framewise.py --geom sav15_101 --reps 0 5 --stride 21
```
