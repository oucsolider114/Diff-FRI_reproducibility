#!/usr/bin/env python
"""Fixed-cadence re-initialization audit (main-text Sec. 4, Supplementary Sec. S24).

The audit re-runs the global initializer every N frames and lets ordinary causal
updates fill the gaps. Four schedules are archived: uninterrupted tracking
(`no_reset`), and resets every 20, 10, and 5 frames.

This script recomputes the published summary **from the per-frame export**
(`data/swellex/cadence_frame_metrics.csv`) and checks the result against the
archived summary (`data/swellex/cadence_summary.csv`). It does not need the
pressure arrays, so it runs from a fresh clone in under a second.

Re-running the estimator itself (which needs `Y_frames.npz`) is a separate,
longer job; see `experiments/rerun_cadence_from_arrays.py`.

Usage:
    python experiments/cadence_audit.py            # writes data/swellex/cadence_audit.csv
    python experiments/cadence_audit.py --stdout   # CSV to stdout
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "swellex"

# published values, used as a regression check on the recomputation
EXPECTED = {
    "no_reset":  dict(fixed_mae=0.0022804084558441405, retained=7,  residual=0.10896777649820387),
    "period20":  dict(fixed_mae=0.0018560122776601158, retained=9,  residual=0.10481745624944859),
    "period10":  dict(fixed_mae=0.0017757209395066976, retained=13, residual=0.10046919927737079),
    "period05":  dict(fixed_mae=0.0021022355955486694, retained=11, residual=0.08874490741409065),
}
ORDER = ["no_reset", "period20", "period10", "period05"]


def recompute():
    df = pd.read_csv(DATA / "cadence_frame_metrics.csv", encoding="utf-8-sig")
    rows = []
    for sched in ORDER:
        d = df[df["schedule"] == sched].sort_values("frame")
        # frame 2 carries the mandatory initialization and is excluded from the
        # scored set; its cost is still part of the runtime total.
        scored = d[d["frame"] > 2]
        rows.append(dict(
            schedule=sched,
            scored_frames=len(scored),
            fixed_mae=scored["fixed_mae"].mean(),
            unique_mae=scored["unique_mae"].mean(),
            retained=int(scored["retained"].sum()),
            residual=scored["residual"].mean(),
            runtime=d["runtime"].sum(),
        ))
    return pd.DataFrame(rows)


def check(out):
    bad = []
    for _, r in out.iterrows():
        e = EXPECTED[r["schedule"]]
        if abs(r["fixed_mae"] - e["fixed_mae"]) > 1e-9:
            bad.append(f"{r['schedule']}: fixed_mae")
        if r["retained"] != e["retained"]:
            bad.append(f"{r['schedule']}: retained")
        if abs(r["residual"] - e["residual"]) > 1e-8:
            bad.append(f"{r['schedule']}: residual")
    return bad


def main():
    out = recompute()
    bad = check(out)

    if "--stdout" in sys.argv:
        out.to_csv(sys.stdout, index=False)
    else:
        dst = DATA / "cadence_audit.csv"
        out.to_csv(dst, index=False)
        print(f"wrote {dst}")

    print(out.to_string(index=False), file=sys.stderr)
    if bad:
        print("REGRESSION FAILED: " + "; ".join(bad), file=sys.stderr)
        sys.exit(1)
    print("all four schedules match the archived summary", file=sys.stderr)


if __name__ == "__main__":
    main()
