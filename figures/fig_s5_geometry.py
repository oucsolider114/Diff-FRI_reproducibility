"""Supplementary Fig. S5: SMT sensitivity to the receiver mask.

Redrawn from data/vla_geometry/csv/trial_summary.csv. Uses the current
terminology (SMT, wavenumber RMSE) rather than the legacy labels.
"""
from pathlib import Path
import csv
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path(__file__).resolve().parents[1]
CSV = BASE / "data/vla_geometry/csv/trial_summary.csv"
OUT = Path(r"E:/Users/lenovo/_dsh_work/figcheck/Supp_FigS5_geometry_tracking_rmse.pdf")
OUT.parent.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(CSV, encoding="utf-8-sig")))
by = collections.defaultdict(list)
for r in rows:
    by[r["variant"]].append(float(r["csmt_tracking_only_mean_rmse_rad_m"]))

order = ["full_40", "sparse_fullspan_20", "short_mid_20", "short_mid_14"]
labels = {
    "full_40": "full_40\n(40 el, 39 m)",
    "sparse_fullspan_20": "sparse_fullspan_20\n(20 el, 39 m)",
    "short_mid_20": "short_mid_20\n(20 el, 19 m)",
    "short_mid_14": "short_mid_14\n(14 el, 13 m)",
}
x = np.arange(len(order))

fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.axvspan(-0.5, 0.5, color="#eef3fb", zorder=0)
ax.axvspan(0.5, 3.5, color="#fdeeee", zorder=0)
for i, v in enumerate(order):
    y = np.array(by[v]) * 1e3
    jitter = (np.random.RandomState(i).rand(len(y)) - 0.5) * 0.16
    ax.scatter(i + jitter, y, s=18, color="0.45", alpha=0.75, linewidths=0, zorder=2)
mean = [np.mean(by[v]) * 1e3 for v in order]
ax.plot(x, mean, "-D", color="#1f5fa9", markersize=6, lw=1.4, label="across-trial mean", zorder=3)
ax.set_xticks(x)
ax.set_xticklabels([labels[v] for v in order], fontsize=8.5)
ax.set_ylabel(r"Mean update-frame wavenumber RMSE ($10^{-3}$ rad/m)", fontsize=9.5)
ax.set_title("SMT sequence sensitivity to the receiver mask", fontsize=10.5)
ax.text(0.20, 0.94, "receiver count halved,\naperture held", transform=ax.transAxes,
        fontsize=8.5, color="#1f5fa9", ha="center")
ax.text(0.74, 0.94, "shortened-aperture cases", transform=ax.transAxes,
        fontsize=8.5, color="#a13a3a", ha="center")
ax.legend(loc="lower right", fontsize=8.5, frameon=False)
ax.grid(axis="y", ls=":", lw=0.6, color="0.8")
ax.set_ylim(-0.002, max(max(by[v]) for v in order) * 1e3 * 1.12)
for sp in ("top", "right"):
    ax.spines[sp].set_visible(False)
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT)
print("means (1e-3 rad/m):", [round(m, 4) for m in mean])
