"""S29c: complete E1 (alpha_t, alpha_t*, their correlation) and E2 on the S28 archive.

E1 uses the archived S3 trajectory (per-frame estimates and KRAKEN references).
E2 uses the archived KRAKEN forward arrays of the S28 audit: no new forward run,
no synthetic pressures. Two quantities are formed there:
  g      = min_xi1 max_m |k_m(xi1) - k_m^ref|          (representability floor)
  eps    = || (I - P) Phi_ref ||_F / || Phi_ref ||_F   (subspace mismatch of the
           KRAKEN receiver basis against the trial basis of the same dimension)
Both are evaluated for the one-parameter chart and for a free three-wavenumber
fit, so the gain of the extra coordinates is measured without observations.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

REPO = Path(r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo")
S28 = Path(r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\需求清单\Diff-FRI_reproducibility\data\fig1_evolution\kraken_timevarying_csmt_crosscheck_formal\arrays\kraken_dynamic_truth.npz")
sys.path.insert(0, str(REPO / "experiments"))
import s29_ocms_manifold_geometry as geo  # noqa: E402

K = 3


def project_frac(pts, target):
    d = np.linalg.norm(pts - target[None, :], axis=1)
    i = int(np.argmin(d))
    return i, float(d[i])


def tangent_at(pts, i):
    if not (0 < i < len(pts) - 1):
        return None
    g = pts[i + 1] - pts[i - 1]
    if np.linalg.norm(pts[i + 1] - pts[i]) + np.linalg.norm(pts[i] - pts[i - 1]) > 0.1:
        return None
    n = np.linalg.norm(g)
    return g / n if n else None


def off_axis(vec, unit):
    return float(np.linalg.norm(vec - (vec @ unit) * unit) / max(np.linalg.norm(vec), 1e-30))


def e1():
    rows = list(csv.DictReader((REPO / "data/synthetic/frame_metrics.csv").open(encoding="utf-8-sig")))
    frames = sorted({int(r["time_index"]) for r in rows})
    reps = sorted({int(r["repeat"]) for r in rows})
    get = {(int(r["repeat"]), int(r["time_index"])): r for r in rows}
    kref = np.array([[float(get[(0, t)][f"kraken_k{m}"]) for m in (1, 2, 3)] for t in frames])
    est = {rep: np.array([[float(get[(rep, t)][f"csmt_k{m}"]) for m in (1, 2, 3)] for t in frames])
           for rep in reps}

    a_truth, a_track, mag = [], [], []
    for j in range(1, len(frames)):
        t = frames[j]
        f = t / (len(frames) - 1)
        c, _ = geo.make_ssp(f)
        lo, hi = 0.95 * geo.OMEGA / c.max(), 0.999999 * geo.OMEGA / c.min()
        _, pts = geo.manifold(c, lo, hi, n_grid=500)
        i, _ = project_frac(pts, kref[j])
        g = tangent_at(pts, i)
        dtruth = kref[j] - kref[j - 1]
        mag.append(float(np.linalg.norm(dtruth)))
        a_truth.append(off_axis(dtruth, g) if g is not None else np.nan)
        per_rep = []
        for rep in reps:
            d = est[rep][j] - est[rep][j - 1]
            ie, _ = project_frac(pts, est[rep][j])
            gg = tangent_at(pts, ie)
            if gg is not None:
                per_rep.append(off_axis(d, gg))
        a_track.append(float(np.median(per_rep)) if per_rep else np.nan)

    a_truth = np.array(a_truth); a_track = np.array(a_track); mag = np.array(mag)
    ok = np.isfinite(a_truth) & np.isfinite(a_track)
    rho, p = spearmanr(a_truth[ok], a_track[ok])
    rho2, p2 = spearmanr(a_truth[ok], mag[ok])
    out = dict(
        frames=int(ok.sum()),
        alpha_truth_median=float(np.median(a_truth[ok])),
        alpha_truth_iqr=[float(np.quantile(a_truth[ok], 0.25)), float(np.quantile(a_truth[ok], 0.75))],
        alpha_track_median=float(np.median(a_track[ok])),
        alpha_track_iqr=[float(np.quantile(a_track[ok], 0.25)), float(np.quantile(a_track[ok], 0.75))],
        spearman_truth_vs_track=float(rho), spearman_p=float(p),
        spearman_truth_vs_step_magnitude=float(rho2), spearman_p2=float(p2),
    )
    return out, a_truth, a_track


def trial_basis(ks, c):
    P = geo.mode_matrix(np.asarray(ks, float), c)
    return P[:, np.arange(1, 41)] .T          # (40, 3)


def subspace_mismatch(Phi_ref, c, ks):
    Phi = trial_basis(np.sort(ks)[::-1], c)
    n = np.linalg.norm(Phi, axis=0)
    Phi = Phi / np.maximum(n, 1e-30)
    Q, _ = np.linalg.qr(Phi)
    R = Phi_ref - Q @ (Q.conj().T @ Phi_ref)
    return float(np.linalg.norm(R) / np.linalg.norm(Phi_ref))


def e2():
    d = np.load(S28, allow_pickle=False)
    ssp = np.asarray(d["ssp_m_s"]); truth = np.asarray(d["truth_k_rad_m"])
    basis = np.asarray(d["kraken_receiver_basis"])
    rows = []
    for t in range(ssp.shape[0]):
        c = ssp[t]
        lo, hi = 0.95 * geo.OMEGA / c.max(), 0.999999 * geo.OMEGA / c.min()
        _, pts = geo.manifold(c, lo, hi, n_grid=500)
        dist = np.abs(pts - truth[t][None, :]).max(axis=1)
        i = int(np.argmin(dist))
        g = float(dist[i])
        Phi_ref = basis[t]
        Phi_ref = Phi_ref / np.linalg.norm(Phi_ref, axis=0)
        eps_chart = subspace_mismatch(Phi_ref, c, pts[i])
        # free three-wavenumber fit of the same subspace
        k = truth[t].copy()
        best = subspace_mismatch(Phi_ref, c, k)
        for scale in (4e-3, 1e-3, 3e-4):
            for _ in range(6):
                for m in range(K):
                    for dx in np.linspace(-scale, scale, 9):
                        cand = k.copy(); cand[m] += dx
                        e = subspace_mismatch(Phi_ref, c, cand)
                        if e < best:
                            best, k = e, cand
        rows.append(dict(frame=t, g=g, eps_chart=eps_chart, eps_free=best,
                         k1=float(truth[t, 0]), k2=float(truth[t, 1]), k3=float(truth[t, 2]),
                         chart_k1=float(pts[i][0]), chart_k2=float(pts[i][1]), chart_k3=float(pts[i][2])))
    g = np.array([r["g"] for r in rows]); ec = np.array([r["eps_chart"] for r in rows])
    ef = np.array([r["eps_free"] for r in rows])
    return dict(frames=len(rows), g_median=float(np.median(g)), g_max=float(g.max()),
                eps_chart_median=float(np.median(ec)), eps_free_median=float(np.median(ef)),
                eps_gain_median=float(np.median(ec - ef)),
                eps_gain_ratio=float(np.median(ec / np.maximum(ef, 1e-12)))), rows


if __name__ == "__main__":
    s1, at, ak = e1()
    print("E1", json.dumps(s1, indent=2))
    s2, rows2 = e2()
    print("E2", json.dumps(s2, indent=2))
    out = Path(os.environ.get("S29_OUT", r"E:\Users\lenovo\_dsh_work\s29_ocms_manifold_geometry"))
    out.mkdir(parents=True, exist_ok=True)
    with (out / "s29c_e1_e2.json").open("w", encoding="utf-8") as fh:
        json.dump(dict(E1=s1, E2=s2), fh, indent=2)
    with (out / "s29c_e2_frames.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows2[0].keys())); w.writeheader(); w.writerows(rows2)
    print("wrote", out)
