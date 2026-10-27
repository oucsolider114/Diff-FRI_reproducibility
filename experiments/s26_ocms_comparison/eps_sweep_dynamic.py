"""OCMS tolerance sensitivity on the evolving-SSP first frame."""
import sys, json
sys.path.insert(0, r"E:\Users\lenovo\_dsh_work")
import numpy as np
import s26_rerun as S
DYN = S.DYN
cfg = dict(json.load(open(r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo\configs\dynamic_local_basis_coupling_suite.json", encoding="utf-8"))["ssp_dynamic"])
cfg["amplitude_m"] = 4.0
Z = S.Z
ssp0, _ = DYN.make_dynamic_ssp(Z, 0.0, cfg)
k_true = S.solve_all_wavenumbers(ssp0, S.FREQ, S.DZ, max_modes=S.ORDERS[-1] + 1)
k_use = np.array([k_true[o - 1] for o in S.ORDERS])
# 用一个固定 realization 的首帧
omega = 2 * np.pi * S.FREQ
Phi0 = np.column_stack([S.solve_mode(float(k), omega, ssp0, S.DZ) for k in k_use])
frames, cleans = S.realize_frames([(ssp0, k_use, Phi0)], 20260801)
Y0 = frames[0][0]
base = [float(np.linalg.norm(Y0[:, l] - cleans[0][:, l])) for l in range(Y0.shape[1])]
print("真实噪声范数 (前3):", [f"{b:.4e}" for b in base[:3]])
facs = [0.7, 0.85, 1.0, 1.15, 1.3]
res = []
for f in facs:
    eps_l = [f * b for b in base]
    try:
        k = S.ocms_estimate_multi(Y0, ssp0, eps_l)
        k = np.asarray(k, dtype=float).ravel()
        if k.size != k_use.size:
            print(f"  容差 x{f}: 返回 {k.size} 个模态（期望 {k_use.size}）-> 视为失败", flush=True)
            res.append(np.nan); continue
        k = np.sort(k)[::-1]
        r = float(np.sqrt(np.mean((k - k_use) ** 2)))
        res.append(r)
        print(f"  容差 x{f}: RMSE {r:.4e} rad/m", flush=True)
    except Exception as e:
        print(f"  容差 x{f}: 异常 {type(e).__name__}: {e}", flush=True)
        res.append(np.nan)
res = np.array(res, dtype=float)
print("\n各因子 RMSE:", [f"{x:.3e}" if np.isfinite(x) else "FAIL" for x in res])
if np.sum(np.isfinite(res)) > 1:
    v = res[np.isfinite(res)]
    print(f"敏感因子 max/min = {v.max()/v.min():.1f}x  (uniform 版正文报告为 267x)")
else:
    print("有效结果不足，无法给出敏感因子")
print("各因子 RMSE:", [f"{x:.3e}" for x in res])
