#!/usr/bin/env python
"""Fig. 2: synthetic accuracy — (a) trajectories, (b) 20 paired realizations.

Reads data/synthetic/frame_metrics.csv and data/synthetic/repeat_summary.csv.

Usage:
    python figures/fig2_accuracy.py            # writes figures/out/fig2_accuracy.pdf
    python figures/fig2_accuracy.py --stdout
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
DATA = ROOT / "data" / "synthetic"
OUT = ROOT / "figures" / "out"


def build():
    fm = pd.read_csv(DATA / "frame_metrics.csv", encoding="utf-8-sig")
    rs = pd.read_csv(DATA / "repeat_summary.csv", encoding="utf-8-sig")

    fig, axs = plt.subplots(2, 1, figsize=(3.45, 4.9),
                            gridspec_kw={"height_ratios": [1.12, 1]})

    # (a) trajectories, first realization
    ax = axs[0]
    r0 = fm[fm["repeat"] == 0].sort_values("time_index")
    t = r0["time_s"].values / 60.0
    for i in (1, 2, 3):
        ax.plot(t, r0[f"kraken_k{i}"].values, "-", lw=1.0, color="black",
                label="KRAKEN" if i == 1 else None)
        ax.plot(t, r0[f"csmt_k{i}"].values, "o", ms=2.5, mfc="white", mec="black",
                mew=0.6, label="CSMT" if i == 1 else None)
    ax.set_xlabel("Time (min)", labelpad=1.5)
    ax.set_ylabel(r"Wavenumber (rad m$^{-1}$)", labelpad=1.5)
    ax.set_title("(a) modal trajectories", fontsize=7.5, loc="left", pad=2)
    ax.set_ylim(1.2245, 1.2670)
    ax.legend(frameon=False, fontsize=6.1, loc="upper center", ncol=2,
              handlelength=1.3, columnspacing=1.4, borderaxespad=0.15)
    ax.ticklabel_format(axis="y", useOffset=False, style="plain")

    # (b) paired RMSE
    ax = axs[1]
    x = rs["full_overall_rmse_rad_m"].values * 1e4
    y = rs["csmt_overall_rmse_rad_m"].values * 1e4
    lim = max(12, x.max() * 1.12)
    ax.plot([0, lim], [0, lim], "--", lw=0.7, color="0.45")
    ax.plot(x, y, "o", ms=3.8, color="0.35", alpha=0.50, mec="none")
    ax.plot([x.mean()], [y.mean()], "D", ms=6.0, color="crimson",
            mec="black", mew=0.6, zorder=5)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel(r"Framewise Full RMSE ($\times10^{-4}$ rad m$^{-1}$)", labelpad=1.5)
    ax.set_ylabel(r"CSMT RMSE ($\times10^{-4}$ rad m$^{-1}$)", labelpad=1.5)
    ax.set_title("(b) 20 paired realizations", fontsize=7.5, loc="left", pad=2)
    ax.text(0.03, 0.96, f"mean ratio {x.mean()/y.mean():.2f}\n95% CI [2.59, 3.69]",
            transform=ax.transAxes, ha="left", va="top", fontsize=6.1, linespacing=1.25)

    fig.tight_layout(pad=0.4)
    return fig, x.mean() / y.mean()


def main():
    fig, ratio = build()
    if "--stdout" in sys.argv:
        fig.savefig(sys.stdout.buffer, format="pdf")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "fig2_accuracy.pdf")
        print(f"wrote {OUT / 'fig2_accuracy.pdf'}")
    print(f"mean ratio = {ratio:.3f}", file=sys.stderr)
    plt.close(fig)


if __name__ == "__main__":
    main()
