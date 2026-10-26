#!/usr/bin/env python
"""Fig. 3: SWellEx-96 Event S5 — (a) residual over 67 updates, (b) accumulated cost.

Reads data/swellex/csmt_fixed_ssp_i9606.csv (trace) and
data/swellex/Table_final_experimental_summary.csv (timing medians).
No timing value is hard-coded: both medians are read from the table.

Usage:
    python figures/fig3_swellex.py            # writes figures/out/fig3_swellex.pdf
    python figures/fig3_swellex.py --stdout
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "swellex"
OUT = ROOT / "figures" / "out"


def build():
    df = pd.read_csv(DATA / "csmt_fixed_ssp_i9606.csv", encoding="utf-8-sig")
    df = df[df["state"].astype(str).str.upper().str.contains("TRACK", na=False)]
    df = df.sort_values("original_frame")
    frame = df["original_frame"].values.astype(float)
    res = df["residual_mean"].values.astype(float)
    roll = pd.Series(res).rolling(7, center=True, min_periods=3).mean().values

    summ = pd.read_csv(DATA / "Table_final_experimental_summary.csv",
                       encoding="utf-8-sig").set_index("Metric")["Value"]
    t_local = float(summ["CSMT runtime median (s)"])
    t_full = float(summ["Representative NNFR runtime median (s)"])

    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.55))

    ax = axs[0]
    # retained-association frames (7 of the 67 scored frames)
    cm = pd.read_csv(DATA / "cadence_frame_metrics.csv", encoding="utf-8-sig")
    nr = cm[(cm["schedule"] == "no_reset") & (cm["retained"] == 1) & (cm["frame"] > 2)]
    kept = sorted(nr["frame"].astype(int).tolist())
    for f in kept:
        ax.axvspan(f - 0.5, f + 0.5, color="0.86", lw=0, zorder=0)

    ax.plot(frame, res, "-o", lw=0.7, ms=2.2, mfc="white", mec="black",
            mew=0.55, color="0.55", label="per frame")
    ax.plot(frame, roll, "-", lw=1.4, color="black", label="7-frame mean")
    ax.set_xlabel("SWellEx frame index", labelpad=1.5)
    ax.set_ylabel("Normalized residual (dimensionless)", labelpad=1.5)
    ax.set_title("(a) projection residual, 67 updates (shaded: 7 retained)", fontsize=7.2, loc="left", pad=2)
    ax.legend(frameon=False, fontsize=6.1, loc="upper left", handlelength=1.6,
              borderaxespad=0.2)

    ax = axs[1]
    n = np.arange(1, len(frame) + 1)
    ax.semilogy(n, n * t_full, "--", lw=1.0, color="black", label="global re-estimate")
    ax.semilogy(n, n * t_local, "-o", lw=1.0, ms=2.0, mfc="white", mec="black",
                mew=0.55, label="CSMT update")
    ax.set_xlabel("Update index", labelpad=1.5)
    ax.set_ylabel("Cumulative time (s, log scale)", labelpad=1.5)
    ax.set_title("(b) accumulated cost (log scale)", fontsize=7.5, loc="left", pad=2)
    ax.legend(frameon=False, fontsize=6.1, loc="lower right", handlelength=1.6,
              borderaxespad=0.2)

    fig.tight_layout(pad=0.4)
    return fig, len(frame), n[-1] * t_local, n[-1] * t_full


def main():
    fig, k, csmt, full = build()
    if "--stdout" in sys.argv:
        fig.savefig(sys.stdout.buffer, format="pdf")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "fig3_swellex.pdf")
        print(f"wrote {OUT / 'fig3_swellex.pdf'}")
    print(f"n={k} csmt={csmt:.2f}s global={full:.1f}s ratio={full/csmt:.0f}x",
          file=sys.stderr)
    plt.close(fig)


if __name__ == "__main__":
    main()
