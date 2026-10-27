"""Sweep the l1 tolerance eps for OCMS and record the single-frame error.

The x-axis is eps / ||noise||_2, where the noise norm is measured from the
clean-minus-noisy difference on the same observation frame.  Diff-FRI is run
once on the same frame as a horizontal reference.
"""
import sys, time, csv
import numpy as np

sys.path[:0] = [r"E:\Users\lenovo\_pdf_extract"]

from ocms_csmt_compare import (ocms_estimate, build_frame, dynamic_ssp,
                               H, C0, FREQ, DZ, Z, K, RECEIVERS, N_SNAPSHOTS)
from mode_fri import direct_estimator

RATIOS = [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5]
SEED = 20260801
OUT = r"E:\Users\lenovo\_pdf_extract\eps_sweep.csv"


def main():
    ssp0, _ = dynamic_ssp(0.0)
    Y, k_true, Yclean = build_frame(ssp0, SEED)
    eps_true = float(np.linalg.norm(Y[:, 0] - Yclean[:, 0]))
    print(f"|p| = {np.linalg.norm(Y[:, 0]):.1f} | exact noise norm = {eps_true:.2f}", flush=True)
    print("true k:", np.round(k_true, 6), flush=True)

    rows = []
    for r in RATIOS:
        eps = r * eps_true
        t0 = time.perf_counter()
        k = ocms_estimate(Y, ssp0, eps=eps)
        dt = time.perf_counter() - t0
        if k is None:
            rmse = float("nan")
            print(f"eps/eps_true={r:.1f}: FAILED ({dt:.0f}s)", flush=True)
        else:
            rmse = float(np.sqrt(np.mean((k - k_true) ** 2)))
            print(f"eps/eps_true={r:.1f}: k={np.round(k, 6)} RMSE={rmse:.3e} ({dt:.0f}s)", flush=True)
        rows.append(dict(ratio=r, eps=eps, ocms_rmse=rmse, seconds=dt))

    t0 = time.perf_counter()
    kf = np.sort(direct_estimator.estimate(
        Y, K, FREQ, RECEIVERS, H, ssp0, seed=20260917))[::-1]
    dtf = time.perf_counter() - t0
    rmse_f = float(np.sqrt(np.mean((kf - k_true) ** 2)))
    print(f"Diff-FRI: k={np.round(kf, 6)} RMSE={rmse_f:.3e} ({dtf:.0f}s)", flush=True)
    rows.append(dict(ratio=float("nan"), eps=float("nan"), ocms_rmse=rmse_f, seconds=dtf))

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
