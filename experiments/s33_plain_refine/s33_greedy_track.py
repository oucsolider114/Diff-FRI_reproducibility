# -*- coding: utf-8 -*-
"""Greedy single-candidate initialization + SMT tracking, same 20 realizations/SSP as nonorth_init."""
import os, sys, csv, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import s26_rerun as S

BASES = S.build_frame_bases()
rows = []
for r in range(20):
    frames, cleans = S.realize_frames(BASES, 20260801 + 1000 * r)
    ssps = [b[0] for b in BASES]
    Y0, k_true0 = frames[0][0], frames[0][1]
    meta = S.DYN.DynamicMetadata(K=S.K, frequency=S.FREQ, array_geometry=S.RECEIVERS.copy(),
                                 water_depth=S.H, SSP=ssps[0].copy())
    t0 = time.perf_counter()
    k_g = S.DYN.dynamic_topn_coarse(Y=Y0, metadata=meta, grid_size=512, top_n=1,
                                    branch_factor=6, beam_width=24,
                                    minimum_separation=0.00066738, dz=S.DZ)
    k_g = np.sort(np.asarray(k_g, float))[::-1]
    dt = time.perf_counter() - t0
    ir = float(np.sqrt(np.mean((k_g - k_true0) ** 2)))
    tr, ok = S.run_chain(k_g, frames, ssps)
    rows.append(dict(realization=r, init_rmse_greedy=ir, track_rmse_greedy=tr,
                     track_ok_greedy=ok, seconds_greedy=dt))
    print(f"[r{r}] init={ir:.4e} track={tr:.4e} ok={ok} {dt:.1f}s", flush=True)

out = r"E:/Users/lenovo/tmp/greedy_track.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", out)
