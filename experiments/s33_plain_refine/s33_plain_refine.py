# -*- coding: utf-8 -*-
"""Compare Diff-FRI vs a plain multi-start grid search + local refinement (Nelder-Mead),
under the same non-orthogonal water-column trial family, on downstream SMT tracking."""
import sys, csv, time
sys.path.insert(0, r"E:\Users\lenovo\_dsh_work")
import numpy as np
import s26_rerun as S
from scipy.optimize import minimize
from mode_fri import direct_estimator

BASES = S.build_frame_bases()
OMEGA = 2 * np.pi * S.FREQ
_calls = {"n": 0}

def residual_norm(k, Y, ssp):
    _calls["n"] += 1
    Phi = np.column_stack([S.solve_mode(float(kk), OMEGA, ssp, S.DZ) for kk in k])
    P = Phi[S.RIDX]
    C, *_ = np.linalg.lstsq(P, Y, rcond=None)
    R = Y - P @ C
    return float(np.mean(np.sum(np.abs(R)**2, axis=0)) / np.mean(np.sum(np.abs(Y)**2, axis=0)))

def simple_refine(Y, k0, ssp):
    res = minimize(lambda x: residual_norm(np.sort(x)[::-1], Y, ssp), k0, method="Nelder-Mead",
                   options=dict(maxiter=400, xatol=1e-6, fatol=1e-12))
    return np.sort(res.x)[::-1]

R_LIST = [int(x) for x in sys.argv[1:]] or list(range(3))
rows = []
for r in R_LIST:
    t_all = time.perf_counter()
    frames, cleans = S.realize_frames(BASES, 20260801 + 1000 * r)
    ssps = [b[0] for b in BASES]
    Y0, k_true0 = frames[0][0], frames[0][1]
    meta = S.DYN.DynamicMetadata(K=S.K, frequency=S.FREQ, array_geometry=S.RECEIVERS.copy(),
                                 water_depth=S.H, SSP=ssps[0].copy())
    t0 = time.perf_counter()
    k_coarse = S.DYN.dynamic_topn_coarse(Y=Y0, metadata=meta, grid_size=512, top_n=5,
                                         branch_factor=6, beam_width=24,
                                         minimum_separation=0.00066738, dz=S.DZ)
    k_coarse = np.sort(np.asarray(k_coarse, float))[::-1]
    t_c = time.perf_counter() - t0
    t0 = time.perf_counter()
    k_fri = direct_estimator.estimate_from_coarse(Y0, S.K, S.FREQ, S.RECEIVERS, S.H, ssps[0],
                                                  int(20260917 + 1000 * r), k_coarse)
    k_fri = np.sort(np.asarray(k_fri, float))[::-1]
    t_f = time.perf_counter() - t0
    _calls["n"] = 0
    t0 = time.perf_counter()
    k_sim = simple_refine(Y0, k_coarse, ssps[0])
    t_s = time.perf_counter() - t0
    ncalls = _calls["n"]
    rmse = lambda k: float(np.sqrt(np.mean((k - k_true0) ** 2)))
    tr_c, ok_c = S.run_chain(k_coarse, frames, ssps)
    tr_f, ok_f = S.run_chain(k_fri, frames, ssps)
    tr_s, ok_s = S.run_chain(k_sim, frames, ssps)
    rows.append(dict(realization=r, init_rmse_coarse=rmse(k_coarse), init_rmse_fri=rmse(k_fri),
                     init_rmse_simple=rmse(k_sim), track_rmse_coarse=tr_c, track_ok_coarse=ok_c,
                     track_rmse_fri=tr_f, track_ok_fri=ok_f, track_rmse_simple=tr_s, track_ok_simple=ok_s,
                     seconds_coarse=t_c, seconds_fri=t_f, seconds_simple=t_s, nm_calls=ncalls))
    print(f"[r{r}] init coarse={rmse(k_coarse):.4e} fri={rmse(k_fri):.4e} simple={rmse(k_sim):.4e} | "
          f"track coarse={tr_c:.4e} fri={tr_f:.4e} simple={tr_s:.4e} | "
          f"{t_c:.1f}+{t_f:.1f}+{t_s:.1f}s ({ncalls} evals, total {time.perf_counter()-t_all:.0f}s)", flush=True)

out = r"E:/Users/lenovo/tmp/nonorth_init.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", out)
