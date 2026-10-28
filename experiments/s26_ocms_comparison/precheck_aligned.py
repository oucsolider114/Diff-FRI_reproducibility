"""Pre-check: rerun S26/S27 on the TRUE evolving SSP (canonical dynamic model).

Does NOT overwrite the archived uniform-water-column results; writes only
*_precheck_dynamic.csv. Answers six questions from the review.
"""
import sys, csv, json, time, os
sys.path.insert(0, r"E:\Users\lenovo\_dsh_work")
import numpy as np
import s26_rerun as S

# ---- Alignment for the S28 audit (option A) ----
# S3 uses grid_min_separation_rad_m = 0.004 from configs/dynamic_local_basis_coupling_suite.json.
# The archived precheck used 0.00066738, which shrinks the 512-point search span from
# 2.05 rad/m to 0.34 rad/m and is the suspected cause of the 30-50x initialization gap.
S.CFG_FRI = dict(S.CFG_FRI)
S.CFG_FRI["grid_min_separation_rad_m"] = 0.004
print("ALIGNED CFG_FRI:", S.CFG_FRI, flush=True)

DYN = S.DYN
CFG_PATH = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo\configs\dynamic_local_basis_coupling_suite.json"
cfg_full = json.load(open(CFG_PATH, encoding="utf-8"))
cfg_dyn = cfg_full["ssp_dynamic"]
cfg_dyn = dict(cfg_dyn)
cfg_dyn["amplitude_m"] = 4.0  # canonical override
Z, N = S.Z, S.N_FRAMES
print("ssp_dynamic:", {k: cfg_dyn[k] for k in ("amplitude_m", "center_depth_m", "vertical_taper_sigma_m")})
print("reference_profile:", cfg_dyn["reference_profile"], flush=True)

def build_dynamic_bases():
    bases = []
    for t in range(N):
        ssp_t, eta = DYN.make_dynamic_ssp(Z, t / (N - 1), cfg_dyn)
        k_true = S.solve_all_wavenumbers(ssp_t, S.FREQ, S.DZ, max_modes=S.ORDERS[-1] + 1)
        k_use = np.array([k_true[o - 1] for o in S.ORDERS])
        omega = 2 * np.pi * S.FREQ
        Phi = np.column_stack([S.solve_mode(float(k), omega, ssp_t, S.DZ) for k in k_use])
        bases.append((ssp_t, k_use, Phi))
    return bases

from mode_fri import direct_estimator
BASES = build_dynamic_bases()
ssps = [b[0] for b in BASES]
print("SSP 跨度: c_min %.2f -> %.2f | k_true frame0 %s frame10 %s" % (
    min(s.min() for s in ssps), max(s.max() for s in ssps),
    np.array2string(BASES[0][1], precision=8), np.array2string(BASES[10][1], precision=8)), flush=True)
print("帧间参考波数差: " + " ".join(f"|f0-f{t}|={np.max(np.abs(BASES[t][1]-BASES[0][1])):.3e}" for t in (1,5,10,15,20)), flush=True)
print("eta 轨迹: " + " ".join(f"{DYN.make_dynamic_ssp(Z, i/(N-1), cfg_dyn)[1]:.3f}" for i in range(N)), flush=True)

rows26, rows27 = [], []
for r in range(20):
    seed = 20260801 + 1000 * r
    frames, cleans = S.realize_frames(BASES, seed)
    Y0, k_true0 = frames[0][0], frames[0][1]
    meta = DYN.DynamicMetadata(K=S.K, frequency=S.FREQ, array_geometry=S.RECEIVERS.copy(),
                                water_depth=S.H, SSP=ssps[0].copy())
    fs = 20260917 + 1000 * r
    t0 = time.perf_counter()
    k_ref, k_coarse, acc, _, _ = DYN.dynamic_topn_refined(Y0, meta, fs, S.CFG_FRI, S.DZ)
    k_ref = np.sort(np.asarray(k_ref, float))[::-1]
    k_coarse = np.sort(np.asarray(k_coarse, float))[::-1]
    dt_fri = time.perf_counter() - t0
    # OCMS (multi-snapshot) on the same frame
    eps_l = [float(np.linalg.norm(Y0[:, l] - cleans[0][:, l])) for l in range(Y0.shape[1])]
    t0 = time.perf_counter()
    k_ocms = S.ocms_estimate_multi(Y0, ssps[0], eps_l)
    dt_ocms = time.perf_counter() - t0
    k_ocms = np.sort(np.asarray(k_ocms, float))[::-1]
    ic_o = float(np.sqrt(np.mean((k_ocms - k_true0) ** 2)))
    ic_f = float(np.sqrt(np.mean((k_ref - k_true0) ** 2)))
    tr_o, _ = S.run_chain(k_ocms, frames, ssps)
    tr_f, _ = S.run_chain(k_ref, frames, ssps)
    # forced refinement (bypass gate)
    k_for = direct_estimator.estimate_from_coarse(Y0, S.K, S.FREQ, S.RECEIVERS, S.H, ssps[0], int(fs), k_coarse)
    k_for = np.sort(np.asarray(k_for, float))[::-1]
    ic_for = float(np.sqrt(np.mean((k_for - k_true0) ** 2)))
    tr_for, _ = S.run_chain(k_for, frames, ssps)
    rows26.append(dict(realization=r, ocms_init_rmse=ic_o, fri_init_rmse=ic_f, gate_accepted=int(acc),
                       ocms_track_rmse=tr_o, fri_track_rmse=tr_f, fri_init_seconds=dt_fri, ocms_init_seconds=dt_ocms))
    rows27.append(dict(realization=r, init_rmse_coarse=ic_f, init_rmse_refined=ic_for,
                       track_rmse_coarse=tr_f, track_rmse_refined=tr_for))
    print(f"[r{r}] init ocms={ic_o:.4e} fri={ic_f:.4e} forced={ic_for:.4e} | track ocms={tr_o:.4e} fri={tr_f:.4e} forced={tr_for:.4e} | gate={int(acc)} | {dt_fri:.1f}+{dt_ocms:.1f}s", flush=True)

for name, rows in [("s26_precheck_aligned.csv", rows26), ("s27_precheck_aligned.csv", rows27)]:
    p = os.path.join(r"E:\Users\lenovo\_dsh_work", name)
    with open(p, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", p, flush=True)
