# Data and license scope

This repository mixes code, derived data, and references to third-party inputs. They do not
carry the same terms, and the single license field of a Zenodo record cannot express that
split. This file states the scope; `LICENSE` carries the MIT text that applies to the code.

## What is covered by which license

| Content | Paths | License |
|---|---|---|
| Source code | `src/`, `experiments/`, `figures/*.py`, `data/*/scripts/*.py` | MIT (see `LICENSE`) |
| Derived numerical data | `data/**/*.csv`, `data/**/*.json`, `data/**/*.npz` | CC BY 4.0 |
| Derived figures | `figures/out/` | CC BY 4.0 |
| Documentation | `README*`, `docs/`, `protocols/` | CC BY 4.0 |

The Zenodo deposit is registered under MIT so that the code is unambiguously reusable. The
derived data listed above are additionally offered under CC BY 4.0; if you reuse them, cite
the article and the archived release as described in `CITATION.cff`.

## Third-party inputs that are not redistributed

| Input | Why it is absent | Where to obtain it |
|---|---|---|
| SWellEx-96 Event S5 raw waveforms | 567 MB, distributed by the source archive | <https://swellex96.ucsd.edu/s5.htm> |
| SAVEX15 temperature and salinity observations | distributed by SEANOE under its own terms | doi:10.17882/72285 |
| KRAKEN executables and source | third-party acoustic model | obtain from the model authors |
| Full three-CTD estimate arrays | restricted by the source archive | retained by the authors |

`data/savex15/arrays/` contains arrays that were **derived** from the SAVEX15 observations
(interpolated sound-speed profiles and the corresponding forward fields). They are not copies
of the SEANOE records and remain subject to the attribution terms above.

## What the archive does not guarantee

The archived NPZ and CSV are authoritative outputs. The scripts kept under
`data/savex15/scripts/` and `data/vla_geometry/scripts/` are retained for provenance; they
expect the original project layout and external solvers, and are not guaranteed to run
standalone. See `docs/REPRODUCE.md` for the three reproduction tiers.
