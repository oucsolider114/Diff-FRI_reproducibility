#!/usr/bin/env python
"""Fig. 1: technical route (no data dependency).

Usage:
    python figures/fig1_flow.py            # writes figures/out/fig1_flow.pdf
    python figures/fig1_flow.py --stdout   # streams the PDF to stdout
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon, FancyArrowPatch

plt.rcParams.update({
    "font.size": 7.5, "axes.linewidth": 0.7,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
})

OUT = Path(__file__).resolve().parents[1] / "figures" / "out"


def build(figsize=(5.75, 3.15)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 14.2); ax.set_ylim(2.30, 10); ax.axis("off")

    def box(x, y, w, h, txt, fs=7.2, bold=False, fc="white", lw=0.9):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                    boxstyle="round,pad=0.05,rounding_size=0.10",
                                    lw=lw, edgecolor="black", facecolor=fc))
        if txt:
            ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
                    fontweight="bold" if bold else "normal", linespacing=1.2)

    def diamond(x, y, w, h, txt, fs=6.3):
        ax.add_patch(Polygon([[x, y + h / 2], [x + w / 2, y], [x, y - h / 2], [x - w / 2, y]],
                             closed=True, lw=0.9, edgecolor="black", facecolor="white"))
        ax.text(x, y, txt, ha="center", va="center", fontsize=fs)

    def arrow(p, q, ls="-", lw=0.8, conn="arc3,rad=0", shA=2, shB=2):
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=lw,
                                     linestyle=ls, color="black",
                                     shrinkA=shA, shrinkB=shB, connectionstyle=conn))

    # input
    ax.text(5, 9.60, r"$\mathbf{Y}_t,\ z_r,\ \mathbf{c}_t$", ha="center", va="center", fontsize=7.4)
    ax.text(5, 9.20, "input", ha="center", va="center", fontsize=6.0, style="italic")
    arrow((5, 9.02), (5, 8.72))

    # module 1
    box(5, 8.15, 6.7, 0.92,
        "Module 1 (shared)\ntrial dictionary  $\\psi\\rightarrow\\Phi\\rightarrow$  residual $J_t(\\mathbf{k})$", fs=6.9)

    # fan out
    arrow((3.3, 7.66), (2.65, 6.58), ls="--", lw=0.7)
    arrow((6.7, 7.66), (7.35, 6.58), ls="--", lw=0.7)
    ax.text(2.30, 7.18, r"$J_t(\mathbf{k})$", fontsize=6.3, ha="right")
    ax.text(7.70, 7.18, r"$J_t(\mathbf{k})$", fontsize=6.3, ha="left")

    # divider labels
    ax.text(4.72, 6.86, r"$t=0$", fontsize=6.1, ha="right")
    ax.text(5.28, 6.86, r"$t\geq 1$", fontsize=6.1, ha="left")

    # module 2
    box(2.6, 4.90, 4.0, 3.30, "", fc="#f7f7f7")
    ax.text(2.6, 6.10, "Module 2\nDiff-FRI, $t=0$", ha="center", va="center",
            fontsize=6.9, fontweight="bold", linespacing=1.10)
    box(2.6, 5.35, 3.3, 0.50, "beam search (coarse)", fs=6.2)
    box(2.6, 4.55, 3.3, 0.50, "continuous refinement", fs=6.2)
    diamond(2.6, 3.68, 2.4, 0.70, "data gate")
    arrow((2.6, 5.86), (2.6, 5.62))
    arrow((2.6, 5.08), (2.6, 4.82))
    arrow((2.6, 4.28), (2.6, 4.05))

    # module 3
    box(7.55, 4.90, 4.10, 3.30, "", fc="#f7f7f7")
    ax.text(7.55, 6.10, "Module 3\nCSMT, $t\\geq 1$", ha="center", va="center",
            fontsize=6.9, fontweight="bold", linespacing=1.10)
    box(7.50, 5.35, 3.00, 0.50, r"prediction $\tilde{k}_t$", fs=6.0)
    box(7.50, 4.55, 3.00, 0.50, r"window $\mathcal{W}$, 9 pts", fs=6.0)
    diamond(7.50, 3.68, 2.60, 0.70, r"mean $\downarrow$ check", fs=5.8)
    arrow((7.55, 5.86), (7.55, 5.62))
    arrow((7.55, 5.08), (7.55, 4.82))
    arrow((7.55, 4.28), (7.55, 4.05))

    # feedback loop
    arrow((8.78, 3.68), (9.01, 4.48), conn="arc3,rad=-0.45", lw=0.7)
    ax.text(9.10, 4.06, "t+1", fontsize=5.6, ha="left", va="center")

    # outputs
    arrow((2.6, 3.33), (2.6, 3.02))
    arrow((7.55, 3.33), (7.55, 3.02))
    ax.text(2.6, 2.80, r"$\hat{\mathbf{k}}_0$", ha="center", va="center", fontsize=7.8)
    ax.text(7.55, 2.80, r"$\hat{\mathbf{k}}_t$", ha="center", va="center", fontsize=7.8)

    # same-gate link
    ax.plot([3.82, 6.18], [3.68, 3.68], ls=(0, (1, 1.6)), lw=0.6, color="0.45")
    ax.text(5.0, 3.94, "shared objective", ha="center", va="center", fontsize=5.3,
            style="italic", bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
    ax.text(10.15, 6.05,
            r"Diff-FRI gate: mean $\downarrow$, worst not $\uparrow$",
            ha="left", va="center", fontsize=5.4, style="italic")
    ax.text(10.15, 5.20, r"window width $\leftarrow \Delta k$, SSP change",
            ha="left", va="center", fontsize=5.6)
    ax.text(10.15, 4.35, r"$J_t$ is produced only by Module 1",
            ha="left", va="center", fontsize=5.3, style="italic")
    ax.plot([5.0, 5.0], [3.20, 6.60], ls=(0, (1, 2)), lw=0.6, color="0.55")
    ax.plot([9.95, 9.95], [3.30, 6.90], ls=(0, (1, 2)), lw=0.55, color="0.60")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig


def main():
    fig = build()
    if "--stdout" in sys.argv:
        fig.savefig(sys.stdout.buffer, format="pdf")
    else:
        OUT.mkdir(parents=True, exist_ok=True)
        fig.savefig(OUT / "fig1_flow.pdf")
        print(f"wrote {OUT / 'fig1_flow.pdf'}")
    plt.close(fig)


if __name__ == "__main__":
    main()
