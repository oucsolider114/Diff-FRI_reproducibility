"""S29d: pressure-domain Delta-r between the one-parameter chart and a free fit.

The S28 audit archived the KRAKEN receiver basis, the reference wavenumbers and the
SSP of its 21 frames, but not the pressure snapshots. Those snapshots are fully
determined by the archived seed schedule of experiments/s26_ocms_comparison/precheck_kraken.py,
so they are regenerated here exactly rather than newly designed.

For every frame and realization the same observations are fitted twice with the same
inverse model (water-column trial basis, per-column normalization, per-snapshot
normalized squared residual J):
  chart : one parameter xi1, the three wavenumbers derived by the closure
  free  : three wavenumbers, coordinate descent from several starts
Reported: J_chart, J_free, Delta = J_chart - J_free, the same quantity relative to
J_free, the wavenumber error of each fit, and J at the reference wavenumbers, which
isolates the shared model mismatch from the closure.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo")
NPZ = Path(r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\需求清单\Diff-FRI_reproducibility\data\fig1_evolution\kraken_timevarying_csmt_crosscheck_formal\arrays\kraken_dynamic_truth.npz")
sys.path.insert(0, str(REPO / "experiments"))
import s29_ocms_manifold_geometry as geo  # noqa: E402

K, L = 3, 20
SNR_DB = 10.0
RIX = np.arange(1, 41)


def coefficients(modes: int, snaps: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.linspace(1.0, 0.55, modes)
    jitter = rng.uniform(0.85, 1.15, size=(modes, snaps))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=(modes, snaps))
    return (base[:, None] * jitter * np.exp(1j * phase)).astype(np.complex128)


def add_noise(clean: np.ndarray, snr_db: float, seed: int):
    rng = np.random.default_rng(seed)
    p_sig = float(np.mean(np.abs(clean) ** 2))
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    noise = (rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape))
    noise *= np.sqrt(p_noise / 2.0)
    return clean + noise


def objective(k, Y, c) -> float:
    Phi = geo.mode_matrix(np.sort(np.asarray(k, float))[::-1], c)[:, RIX].T
    n = np.linalg.norm(Phi, axis=0)
    Phi = Phi / np.maximum(n, 1e-30)
    Q, _ = np.linalg.qr(Phi)
    R = Y - Q @ (Q.conj().T @ Y)
    num = np.sum(np.abs(R) ** 2, axis=0)
    den = np.maximum(np.sum(np.abs(Y) ** 2, axis=0), 1e-30)
    return float(np.mean(num / den))


def chart_fit(Y, c, lo, hi, n=260):
    xi, pts = geo.manifold(c, lo, hi, n_grid=220)
    if len(xi) < 5:
        return None, None
    best_j, best_k = np.inf, None
    for p in pts:
        j = objective(p, Y, c)
        if j < best_j:
            best_j, best_k = j, p.copy()
    # local refine around the best chart point
    step = (hi - lo) / 60.0
    for _ in range(2):
        for x in np.linspace(best_k[0] - step, best_k[0] + step, 15):
            kk = geo.mode_matrix
            # nearest chart point to this xi
            i = int(np.argmin(np.abs(xi - x)))
            cand = pts[i]
            j = objective(cand, Y, c)
            if j < best_j:
                best_j, best_k = j, cand.copy()
        step /= 4.0
    return best_j, best_k


def free_fit(Y, c, starts):
    best_j, best_k = np.inf, None
    for k0 in starts:
        k = np.sort(np.asarray(k0, float))[::-1].copy()
        j = objective(k, Y, c)
        for scale in (6e-3, 2.5e-3, 1e-3, 4e-4):
            for _ in range(4):
                for m in range(K):
                    for dx in np.linspace(-scale, scale, 9):
                        cand = k.copy(); cand[m] += dx
                        jj = objective(cand, Y, c)
                        if jj < j:
                            j, k = jj, cand
        if j < best_j:
            best_j, best_k = j, k
    return best_j, best_k


def main():
    d = np.load(NPZ, allow_pickle=False)
    truth = np.asarray(d["truth_k_rad_m"], float)
    basis = np.asarray(d["kraken_receiver_basis"], complex)
    ssp = np.asarray(d["ssp_m_s"], float)
    rows = []
    for t in range(truth.shape[0]):
        c = ssp[t]
        lo, hi = 0.95 * geo.OMEGA / c.max(), 0.999999 * geo.OMEGA / c.min()
        for r in range(20):
            seed0 = 20260801 + 1000 * r
            C = coefficients(K, L, seed0 + t)
            Y = add_noise(basis[t] @ C, SNR_DB, seed0 + t + 5000)
            j_ref = objective(truth[t], Y, c)
            j_ch, k_ch = chart_fit(Y, c, lo, hi)
            if j_ch is None:
                continue
            j_fr, k_fr = free_fit(Y, c, [k_ch, truth[t], truth[t] * 1.002])
            rows.append(dict(frame=t, rep=r, J_chart=j_ch, J_free=j_fr, J_ref=j_ref,
                             dJ=j_ch - j_fr, dJ_rel=(j_ch - j_fr) / max(j_fr, 1e-30),
                             err_chart_max=float(np.abs(np.sort(k_ch)[::-1] - truth[t]).max()),
                             err_free_max=float(np.abs(np.sort(k_fr)[::-1] - truth[t]).max()),
                             err_ref_max=0.0))
        print("frame", t, "done", flush=True)

    Jc = np.array([x["J_chart"] for x in rows]); Jf = np.array([x["J_free"] for x in rows])
    Jr = np.array([x["J_ref"] for x in rows]); dJ = np.array([x["dJ"] for x in rows])
    ec = np.array([x["err_chart_max"] for x in rows]); ef = np.array([x["err_free_max"] for x in rows])
    s = dict(records=len(rows),
             J_chart_median=float(np.median(Jc)), J_free_median=float(np.median(Jf)),
             J_ref_median=float(np.median(Jr)), dJ_median=float(np.median(dJ)),
             dJ_rel_median=float(np.median(dJ / np.maximum(Jf, 1e-30))),
             dJ_q25=float(np.quantile(dJ, 0.25)), dJ_q75=float(np.quantile(dJ, 0.75)),
             frac_dJ_positive=float(np.mean(dJ > 0)),
             err_chart_median=float(np.median(ec)), err_free_median=float(np.median(ef)),
             noise_energy_fraction=1.0 / (10 ** (SNR_DB / 10.0)))
    print(json.dumps(s, indent=2))
    out = Path(os.environ.get("S29_OUT", str(Path(__file__).resolve().parent / "s29_ocms_manifold_geometry")))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "s29d_pressure_delta.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (out / "s29d_pressure_delta.json").write_text(json.dumps(s, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
