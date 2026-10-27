"""Forced ablation: coarse-grid initialization vs continuous refinement (gate bypassed), same SMT.

In s26_rerun the data-only gate rejected the refinement in all 20 realizations, so the
returned estimate equals the coarse solution. Here the refinement is taken directly from
direct_estimator.estimate_from_coarse, bypassing the gate, so the two arms actually differ.
"""
import sys, csv, time
sys.path.insert(0, r"E:\Users\lenovo\_dsh_work")
import numpy as np
import s26_rerun as S
from mode_fri import direct_estimator

BASES = S.build_frame_bases()
rows = []
for r in range(20):
    coef_seed = 20260801 + 1000 * r
    frames, cleans = S.realize_frames(BASES, coef_seed)
    ssps = [b[0] for b in BASES]
    Y0, k_true0 = frames[0][0], frames[0][1]
    meta = S.DYN.DynamicMetadata(K=S.K, frequency=S.FREQ, array_geometry=S.RECEIVERS.copy(),
                                 water_depth=S.H, SSP=ssps[0].copy())
    seed = 20260917 + 1000 * r
    t0 = time.perf_counter()
    k_coarse = S.DYN.dynamic_topn_coarse(Y=Y0, metadata=meta, grid_size=512, top_n=5,
                                         branch_factor=6, beam_width=24,
                                         minimum_separation=0.00066738, dz=S.DZ)
    t_coarse = time.perf_counter() - t0
    k_coarse = np.sort(np.asarray(k_coarse, float))[::-1]
    t0 = time.perf_counter()
    k_ref = direct_estimator.estimate_from_coarse(Y0, S.K, S.FREQ, S.RECEIVERS, S.H, ssps[0], int(seed), k_coarse)
    t_ref = time.perf_counter() - t0
    k_ref = np.sort(np.asarray(k_ref, float))[::-1]
    ic = float(np.sqrt(np.mean((k_coarse - k_true0) ** 2)))
    ir = float(np.sqrt(np.mean((k_ref - k_true0) ** 2)))
    trc, okc = S.run_chain(k_coarse, frames, ssps)
    trr, okr = S.run_chain(k_ref, frames, ssps)
    rows.append(dict(realization=r, init_rmse_coarse=ic, init_rmse_refined=ir,
                     track_rmse_coarse=trc, track_rmse_refined=trr,
                     seconds_coarse=t_coarse, seconds_refined=t_ref))
    print(f"[r{r}] init coarse={ic:.4e} refined={ir:.4e} | track coarse={trc:.4e} refined={trr:.4e} | {t_coarse:.1f}+{t_ref:.1f}s", flush=True)

out = r"E:\Users\lenovo\_dsh_work\refine_ablation_forced.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", out, flush=True)
