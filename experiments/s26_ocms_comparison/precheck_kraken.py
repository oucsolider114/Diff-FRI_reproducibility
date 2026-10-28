"""A'-1: S28 re-run with KRAKEN receiver basis instead of the water-column basis.

Only the observation model changes: frames are built from
kraken_receiver_basis[t] @ C with KRAKEN truth wavenumbers, keeping the same
coefficients/noise generators, 10 dB, 20 snapshots, 21 frames, and the same
OCMS / Diff-FRI initializers + SMT tracker as precheck_dynamic.py.
"""
import sys, csv, json, time, os
sys.path.insert(0, r"E:\Users\lenovo\_dsh_work")
import numpy as np
import s26_rerun as S

DYN = S.DYN
NPZ = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\需求清单\Diff-FRI_reproducibility\data\fig1_evolution\kraken_timevarying_csmt_crosscheck_formal\arrays\kraken_dynamic_truth.npz"

d = np.load(NPZ, allow_pickle=True)
truth = np.asarray(d["truth_k_rad_m"], float)          # (21,3)
basis = np.asarray(d["kraken_receiver_basis"], complex) # (21,40,3)
ssp = np.asarray(d["ssp_m_s"], float)                  # (21,41)
recdep = np.asarray(d["receiver_depths_m"], float)     # (40,)
zdep = np.asarray(d["depths_m"], float)                # (41,)
print("npz: frames=%d receivers=%d depths=%d" % (truth.shape[0], recdep.size, zdep.size), flush=True)
print("truth frame0 :", np.array2string(truth[0], precision=8), flush=True)
print("truth frame20:", np.array2string(truth[-1], precision=8), flush=True)

# The npz uses a 1-m depth grid (41 points, 0-40 m); s26_rerun defaults to dz=0.125.
# Align the module-level grid so the KRAKEN basis and the tracker see the same dz.
S.DZ = float(zdep[1] - zdep[0])
S.Z = np.asarray(zdep, float).copy()
S.RIDX = np.array([int(np.argmin(np.abs(S.Z - r))) for r in recdep])
print("aligned dz =", S.DZ, "| Z size =", S.Z.size, "| RIDX max =", int(S.RIDX.max()), flush=True)

N = truth.shape[0]
assert N == S.N_FRAMES, (N, S.N_FRAMES)

# S3-compatible receiver ordering: s26_rerun uses linspace(1,40,40)
RECV_OK = np.allclose(recdep, S.RECEIVERS)
print("receiver grid matches S.RECEIVERS:", RECV_OK, flush=True)

# frames: KRAKEN basis @ coefficients + exact-SNR noise (S3 style)
frames, cleans = [], []
for r in range(20):
    fr, cl = [], []
    seed0 = 20260801 + 1000 * r
    for t in range(N):
        C = S.generate_complex_coefficients(S.K, S.N_SNAPSHOTS, seed0 + t)
        clean = basis[t] @ C
        noisy, _ = S.add_complex_white_noise(clean, S.SNR_DB, seed0 + t + 5000)
        fr.append((noisy, truth[t])); cl.append(clean)
    frames.append(fr); cleans.append(cl)

rows26, rows27 = [], []
for r in range(20):
    fr = frames[r]
    Y0, k_true0 = fr[0][0], fr[0][1]
    meta = DYN.DynamicMetadata(K=S.K, frequency=S.FREQ, array_geometry=S.RECEIVERS.copy(),
                               water_depth=S.H, SSP=ssp[0].copy())
    fs = 20260917 + 1000 * r
    t0 = time.perf_counter()
    k_ref, k_coarse, acc, _, _ = DYN.dynamic_topn_refined(Y0, meta, fs, S.CFG_FRI, S.DZ)
    k_ref = np.sort(np.asarray(k_ref, float))[::-1]
    k_coarse = np.sort(np.asarray(k_coarse, float))[::-1]
    dt_fri = time.perf_counter() - t0
    eps_l = [float(np.linalg.norm(Y0[:, l] - cleans[r][0][:, l])) for l in range(Y0.shape[1])]
    t0 = time.perf_counter()
    k_ocms = S.ocms_estimate_multi(Y0, ssp[0], eps_l)
    dt_ocms = time.perf_counter() - t0
    k_ocms = np.sort(np.asarray(k_ocms, float))[::-1]
    ic_o = float(np.sqrt(np.mean((k_ocms - k_true0) ** 2)))
    ic_f = float(np.sqrt(np.mean((k_ref - k_true0) ** 2)))
    tr_o, _ = S.run_chain(k_ocms, fr, list(ssp))
    tr_f, _ = S.run_chain(k_ref, fr, list(ssp))
    rows26.append(dict(realization=r, ocms_init_rmse=ic_o, fri_init_rmse=ic_f, gate_accepted=int(acc),
                       ocms_track_rmse=tr_o, fri_track_rmse=tr_f, fri_init_seconds=dt_fri, ocms_init_seconds=dt_ocms))
    print("[r%d] init ocms=%.4e fri=%.4e | track ocms=%.4e fri=%.4e | gate=%d | %.1f+%.1fs"
          % (r, ic_o, ic_f, tr_o, tr_f, int(acc), dt_fri, dt_ocms), flush=True)

p = r"E:\Users\lenovo\_dsh_work\s26_precheck_kraken.csv"
with open(p, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows26[0])); w.writeheader(); w.writerows(rows26)
print("wrote", p, flush=True)
