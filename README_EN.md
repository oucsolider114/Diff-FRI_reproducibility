# Causal sequential tracking of modal horizontal wavenumbers

Code and data for the JASA Express Letters article

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
python experiments/cadence_audit.py      # writes data/swellex/cadence_audit.csv
bash figures/make_all.sh                 # writes figures/out/*.pdf
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
| `protocols/` | Experiment parameters as JSON |
| `figures/` | Figure scripts and their PDF output |
| `docs/` | Column dictionary and step-by-step reproduction notes |

## Reproducing each figure

| Figure | Command | Reads |
|---|---|---|
| Fig. 1 (route) | `python figures/fig1_flow.py` | no data |
| Fig. 2 (accuracy) | `python figures/fig2_accuracy.py` | `data/synthetic/*.csv` |
| Fig. 3 (field) | `python figures/fig3_swellex.py` | `data/swellex/csmt_fixed_ssp_i9606.csv`, `…/Table_final_experimental_summary.csv` |

Every number printed in the article is either read directly from a CSV in `data/` or
computed from one by a script in `figures/` or `experiments/`. No timing value is
hard-coded: the two medians used in Fig. 3(b) are read from the experimental summary table.

## What is not included

| Item | Reason | Where to get it |
|---|---|---|
| SWellEx-96 raw waveforms | 567 MB, publicly archived | <https://swellex96.ucsd.edu/s5.htm> |
| SAVEX15 environmental data | distributed by SEANOE | doi:10.17882/72285 |
| `.npz` intermediate arrays | large, regenerable from the CSV exports | Zenodo archive (see `CITATION.cff`) |
| KRAKEN executables | third-party, not redistributable | obtain from the authors of the model |
| Full three-CTD estimate arrays | restricted by the source archive | on request from the corresponding author |

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
