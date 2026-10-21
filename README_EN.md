# Causal sequential tracking of modal horizontal wavenumbers

Code and data for the JASA Express Letters article

**Version v1.1.2 (2026-10-21)** — adds the SAVEX15, VLA-geometry, initialization-ablation,
and predictor-holdout datasets. The archive supports recomputation of the main reported
figures and summary statistics from derived data. Selected experiment-level reruns require
external inputs and the original computing environment; see `docs/REPRODUCE.md` for the
three reproduction tiers.

> **Causal sequential tracking of modal horizontal wavenumbers via bounded local updates**
> Lingnan Liu, Fajun Liu, and Pengyu Wang (Ocean University of China)

One global differentiable finite-rate-of-innovation (Diff-FRI) initialization followed by
bounded, sound-speed-conditioned local updates (CSMT). In a 40 m synthetic waveguide the
tracker lowers all-frame wavenumber error by a factor of 3.09 relative to framewise
estimation; on SWellEx-96 Event S5 a local update costs 0.108 s against 102.58 s for a
global re-estimate. Association, not fit, is the operating limit.

## Quick start

```bash
pip install -r requirements.txt
python experiments/cadence_audit.py      # rewrites data/swellex/cadence_audit.csv (manifest-protected)
bash figures/make_all.sh                 # writes figures/out_repro/; archived figures untouched
```

To check a single figure without writing files:

```bash
python figures/fig2_accuracy.py --stdout > /tmp/fig2.pdf
```

## Repository layout

| Path | Contents |
|---|---|
| `src/mode_fri/` | Estimator implementation: trial dictionary, Diff-FRI initialization, CSMT update, configuration |
| `experiments/` | Drivers for the four reported experiments plus the cadence audit |
| `data/synthetic/` | Per-frame and per-realization exports for Sec. 3.1 (Fig. 2) |
| `data/robustness/` | One-factor sweeps for Sec. 3.2 |
| `data/swellex/` | Per-frame exports for Sec. 3.3–3.4 (Fig. 3) and the cadence audit |
| `data/baselines/` | Estimator-class comparison behind Table I |
| `data/arrays/` | SWellEx pressure-field arrays (experiment-level re-run needs the original project layout) |
| `data/savex15/` | Sec. S10: SAVEX15 forward arrays, SSPs, per-realization results, StageA/StageB scripts |
| `data/vla_geometry/` | Sec. S20: controlled VLA-geometry sweep — 1680 per-frame rows, summaries, protocol lock |
| `data/ablation_initialization/` | Sec. S6: three-stage initialization ablation (100 trials) |
| `data/predictor_holdout/` | Sec. S18: independent holdout for the fixed predictor and closed-loop β sweep |
| `protocols/` | Experiment parameters as JSON |
| `figures/` | Figure scripts and their PDF output |
| `docs/` | Column dictionary and step-by-step reproduction notes |

## Reproducing each figure

| Figure | Command | Reads |
|---|---|---|
| Fig. 1 (route) | `python figures/fig1_flow.py` | no data |
| Fig. 2 (accuracy) | `python figures/fig2_accuracy.py` | `data/synthetic/*.csv` |
| Fig. 3 (field) | `python figures/fig3_swellex.py` | `data/swellex/csmt_fixed_ssp_i9606.csv`, `…/Table_final_experimental_summary.csv` |

Nearly every number printed in the article is either read directly from a CSV in `data/` or
computed from one by a script in `figures/` or `experiments/`. The one exception is the
K=3/5/8 scan in Sec. S2, whose per-trial records are not exported separately (noted in S2).
No timing value is hard-coded: the two medians used in Fig. 3(b) are read from the
experimental summary table.

The scripts under `data/savex15/scripts/` and `data/vla_geometry/scripts/` are retained for
provenance and expect the original project layout; see `docs/REPRODUCE.md` (Tier C) before
attempting a cold-start rerun. The archived NPZ/CSV are the authoritative outputs.

## What is not included

| Item | Reason | Where to get it |
|---|---|---|
| SWellEx-96 raw waveforms | 567 MB, publicly archived | <https://swellex96.ucsd.edu/s5.htm> |
| SAVEX15 environmental data | distributed by SEANOE | doi:10.17882/72285 |
| SAVEX15 forward-field KRAKEN output (`.prt`/`.mod`/`.env`) | ~42 MB of intermediates, regenerable by `NN_FRI_SAVEX15_Physics_Repair_v1.py` | this repository ships the resulting `.npz` arrays |
| KRAKEN executables | third-party, not redistributable | obtain from the authors of the model |
| Full three-CTD estimate arrays | restricted by the source archive | on request from the corresponding author |

## v1.1.1 -> v1.1.2

- The corresponding-author e-mail (`wangpengyu@ouc.edu.cn`) is kept, consistent with the Zenodo record page and the article's first page.
- Front-matter reproduction claim narrowed to a two-tier statement (recomputation from derived data vs. experiment-level rerun), consistent with Tier B/C in `docs/REPRODUCE.md`.
- `CITATION.cff` message now asks readers to cite this archived release, noting that the journal article will be citable once published; the software DOI is no longer presented as an article DOI.

## v1.1.0 -> v1.1.1

- All text files normalized to **LF**; added `.gitattributes` (`* text=auto eol=lf`).
- `MANIFEST.sha256` recomputed on LF, so the local tree, the GitHub archive and the
  Zenodo record now agree. The v1.1.0 manifest was computed on CRLF working-tree files
  and therefore reported 136 spurious mismatches against the LF release archive.
- `figures/make_all.sh` writes to `figures/out_repro/` and no longer overwrites the
  archived figures that the manifest protects.
- Tier A scope in `docs/REPRODUCE.md` narrowed; the K=3/5/8 sweep is noted as having no
  per-trial export.

## Data license and third-party terms

Code is released under the MIT license. Derived tables and figures are released under
CC BY 4.0. The original SWellEx-96 and SAVEX15 data remain under their own terms; nothing
in `data/` redistributes raw waveforms.

## Integrity check

`MANIFEST.sha256` lists the SHA-256 of every tracked file. Regenerate or verify with:

```bash
find . -type f -not -path './.git/*' -not -name 'MANIFEST.sha256' \
  -exec sha256sum {} + | sort -k2 > MANIFEST.sha256
sha256sum -c MANIFEST.sha256
```

## Citation

See `CITATION.cff`. Please cite the article and the archived release.

## Contact

Pengyu Wang — wangpengyu@ouc.edu.cn
