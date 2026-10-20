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


def build(figsize=(3.45, 4.75)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 10); ax.set_ylim(0.4, 10); ax.axis("off")

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
        ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=lw,
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
    ax.text(2.6, 6.10, "Module 2\nDiff-FRI\n$t=0$", ha="center", va="center",
            fontsize=6.9, fontweight="bold", linespacing=1.12)
    box(2.6, 5.28, 3.3, 0.50, "beam search (coarse)", fs=6.2)
    box(2.6, 4.64, 3.3, 0.50, "continuous refinement", fs=6.2)
    diamond(2.6, 3.86, 2.4, 0.70, "data gate")
    arrow((2.6, 5.86), (2.6, 5.56))
    arrow((2.6, 5.03), (2.6, 4.90))
    arrow((2.6, 4.39), (2.6, 4.22))

    # module 3
    box(7.55, 4.90, 4.10, 3.30, "", fc="#f7f7f7")
    ax.text(7.55, 6.10, "Module 3\nCSMT\n$t\\geq 1$", ha="center", va="center",
            fontsize=6.9, fontweight="bold", linespacing=1.12)
    box(7.50, 5.28, 3.00, 0.50, r"prediction $\tilde{k}_t$", fs=6.0)
    box(7.50, 4.64, 3.00, 0.50, r"window $\mathcal{W}$, 9 pts", fs=6.0)
    diamond(7.50, 3.86, 2.30, 0.70, "data gate")
    arrow((7.55, 5.86), (7.55, 5.56))
    arrow((7.55, 5.03), (7.55, 4.90))
    arrow((7.55, 4.39), (7.55, 4.22))

    # feedback loop
    arrow((8.66, 3.96), (9.12, 4.58), conn="arc3,rad=0.40", lw=0.7)
    arrow((9.12, 4.58), (9.00, 5.28), lw=0.7)
    ax.text(9.20, 4.58, "t+1", fontsize=5.6, ha="left", va="center")

    # outputs
    arrow((2.6, 3.50), (2.6, 3.02))
    arrow((7.55, 3.50), (7.55, 3.02))
    ax.text(2.6, 2.80, r"$\hat{\mathbf{k}}_0$", ha="center", va="center", fontsize=7.8)
    ax.text(7.55, 2.80, r"$\hat{\mathbf{k}}_t$", ha="center", va="center", fontsize=7.8)

    # same-gate link
    ax.plot([3.82, 6.18], [3.86, 3.86], ls=(0, (1, 1.6)), lw=0.6, color="0.45")
    ax.text(5.0, 4.12, "same gate", ha="center", va="center", fontsize=5.3,
            style="italic", bbox=dict(boxstyle="square,pad=0.12", fc="white", ec="none"))
    ax.text(5.0, 2.05,
            r"accept if mean $\downarrow$ and worst snapshot not $\uparrow$",
            ha="center", va="center", fontsize=5.7, style="italic")
    ax.text(5.0, 1.35, r"window width $\leftarrow \Delta k$, SSP change",
            ha="center", va="center", fontsize=5.9)
    ax.text(5.0, 0.72, r"$J_t$ is produced only by Module 1",
            ha="center", va="center", fontsize=5.5, style="italic")
    ax.plot([5.0, 5.0], [3.20, 6.60], ls=(0, (1, 2)), lw=0.6, color="0.55")

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
