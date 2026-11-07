# S31 — framewise control under a uniform +-1 m/s inverse-SSP offset

## Question

S25 reports that SMT keeps a criterion-met fraction of 0.9875 when the inverse
SSP is offset by +-1 m/s. Does SMT still beat independent framewise estimation
under the same offset, or does the margin vanish?

## Design

- Environment: same 40-m, 300-Hz, 21-frame trajectory and 20 paired
  coefficient/noise realisations as S25 (10 dB, 20 snapshots).
- Observations are synthesised with the truth SSP; the estimator receives
  `ssp +- 1 m/s`.
- Both arms use the same initializer: beam search over a 512-point grid
  (branch 6, beam 24, five candidates) followed by
  `direct_estimator.estimate_from_coarse`, with the acceptance-test
  refinement.
- `smt`: one full initialization at frame 0, then the SMT update.
- `framewise`: one full initialization per frame.

## Results (n = 20 paired realisations)

| delta (m/s) | SMT RMSE | framewise RMSE | ratio | SMT criterion-met | framewise criterion-met | SMT wins |
|---|---|---|---|---|---|---|
| -1 | 9.01e-4 | 1.285e-3 | 1.43 | 0.957 | 0.581 | 20/20 |
| +1 | 9.16e-4 | 1.322e-3 | 1.44 | 0.952 | 0.545 | 20/20 |

Nominal reference (no offset, same archive, tracking frames):
SMT 2.52e-4, framewise 9.18e-4, ratio 3.64; criterion-met 0.991 vs 0.783.

## What it shows

1. The margin narrows but does not vanish: 3.64x nominal to 1.43-1.44x at
   +-1 m/s.
2. SMT wins in every paired realisation at both offsets (20/20), so the
   advantage is not an artefact of averaging.
3. The criterion-met fraction separates the two arms more sharply than the
   RMSE ratio: 0.95 for SMT against 0.55-0.58 for framewise.
4. SMT degrades more in relative terms (2.52e-4 to 9.0e-4, 3.6x) than
   framewise (9.18e-4 to 1.29e-3, 1.4x), because the update window is centred
   on the recent track; it remains ahead in absolute terms because its
   nominal level is lower.

## Limitations

- The SMT RMSE here is 9.01-9.16e-4, against 0.848-0.889e-3 in S25, i.e.
  1.4-8% higher. The coefficient/noise seeds differ and the frame-0
  initializer is re-run rather than reused, so the absolute level is not
  identical; only the paired comparison within this archive is interpreted.
- Only +-1 m/s is tested. At +-2 m/s both arms are near failure in S25
  (criterion-met 0.005-0.0675), so the comparison has no discriminating power.
- Framewise is defined as the full initializer applied to each frame, which
  matches the "independent framewise estimation" of the main text.

## Reproduce

```bash
S31_OUT=./results python s31_ssp_shift_framewise.py --delta -1 --repeats 0 20
S31_OUT=./results python s31_ssp_shift_framewise.py --delta 1  --repeats 0 20
```

Output: `results/s31_pooled.csv` (40 rows, two offsets x 20 realisations).
