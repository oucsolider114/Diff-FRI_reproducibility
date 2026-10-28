"""A'-1 refine ablation on the KRAKEN basis.

Same observation model as precheck_kraken.py (kraken_receiver_basis @ C,
KRAKEN truth, 1-m depth grid). Compares:
  coarse only      -> k_coarse from dynamic_topn_refined
  forced refined   -> estimate_from_coarse (bypasses the acceptance test)
  gated refined    -> k_ref from dynamic_topn_refined (whatever the gate returned)
"""
import sys, csv, time
sys.path.insert(0, r"E:\Users\lenovo\_dsh_work")
import numpy as np
import s26_rerun as S
from mode_fri import direct_estimator

DYN = S.DYN
NPZ = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\需求清单\Diff-FRI_reproducibility\data\fig1_evolution\kraken_timevarying_csmt_crosscheck_formal\arrays\kraken_dynamic_truth.npz"

d = np.load(NPZ, allow_pickle=True)
truth = np.asarray(d["truth_k_rad_m"], float)
basis = np.asarray(d["kraken_receiver_basis"], complex)
ssp = np.asarray(d["ssp_m_s"], float)
recdep = np.asarray(d["receiver_depths_m"], float)
zdep = np.asarray(d["depths_m"], float)

S.DZ = float(zdep[1] - zdep[0])
S.Z = np.asarray(zdep, float).copy()
S.RIDX = np.array([int(np.argmin(np.abs(S.Z - r))) for r in recdep])
N = truth.shape[0]
print("dz =", S.DZ, "| frames =", N, flush=True)

rows = []
for r in range(20):
    seed0 = 20260801 + 1000 * r
    fr = []
    for t in range(N):
        C = S.generate_complex_coefficients(S.K, S.N_SNAPSHOTS, seed0 + t)
        clean = basis[t] @ C
        noisy, _ = S.add_complex_white_noise(clean, S.SNR_DB, seed0 + t + 5000)
        fr.append((noisy, truth[t]))
    Y0, k_true0 = fr[0][0], fr[0][1]
    meta = DYN.DynamicMetadata(K=S.K, frequency=S.FREQ, array_geometry=S.RECEIVERS.copy(),
                               water_depth=S.H, SSP=ssp[0].copy())
    fs = 20260917 + 1000 * r
    k_ref, k_coarse, acc, _, _ = DYN.dynamic_topn_refined(Y0, meta, fs, S.CFG_FRI, S.DZ)
    k_ref = np.sort(np.asarray(k_ref, float))[::-1]
    k_coarse = np.sort(np.asarray(k_coarse, float))[::-1]
    k_for = direct_estimator.estimate_from_coarse(Y0, S.K, S.FREQ, S.RECEIVERS, S.H, ssp[0], int(fs), k_coarse)
    k_for = np.sort(np.asarray(k_for, float))[::-1]
    e = lambda k: float(np.sqrt(np.mean((k - k_true0) ** 2)))
    tr_c, _ = S.run_chain(k_coarse, fr, list(ssp))
    tr_r, _ = S.run_chain(k_ref, fr, list(ssp))
    tr_f, _ = S.run_chain(k_for, fr, list(ssp))
    rows.append(dict(realization=r, gate_accepted=int(acc),
                     init_coarse=e(k_coarse), init_refined=e(k_ref), init_forced=e(k_for),
                     track_coarse=tr_c, track_refined=tr_r, track_forced=tr_f))
    print("[r%d] init coarse=%.3e refined=%.3e forced=%.3e | track coarse=%.3e refined=%.3e forced=%.3e | gate=%d"
          % (r, e(k_coarse), e(k_ref), e(k_for), tr_c, tr_r, tr_f, int(acc)), flush=True)

p = r"E:\Users\lenovo\_dsh_work\s27_precheck_kraken.csv"
with open(p, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
print("wrote", p, flush=True)
