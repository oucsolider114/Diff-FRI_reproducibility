import numpy as np, csv, sys
sys.stdout.reconfigure(encoding="utf-8")

# A'-1 (npz, periodic)
p = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\需求清单\Diff-FRI_reproducibility\data\fig1_evolution\kraken_timevarying_csmt_crosscheck_formal\arrays\kraken_dynamic_truth.npz"
d = np.load(p, allow_pickle=True)
eta = d["thermocline_displacement_m"]
k = d["truth_k_rad_m"]
print("=== A'-1 npz（周期 SSP） ===")
print("  thermocline displacement: min %.3f max %.3f  (span %.3f m)" % (eta.min(), eta.max(), eta.max() - eta.min()))
print("  truth k drift (max-min per mode):", np.array2string(k.max(axis=0) - k.min(axis=0), precision=6))

# S3 main experiment (frame_metrics.csv)
p2 = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo\data\synthetic\frame_metrics.csv"
rows = list(csv.DictReader(open(p2, encoding="utf-8-sig")))
r0 = [r for r in rows if int(r["repeat"]) == 0]
r0.sort(key=lambda r: int(r["time_index"]))
eta2 = [float(r["thermocline_displacement_m"]) for r in r0]
kk = np.array([[float(r["kraken_k1"]), float(r["kraken_k2"]), float(r["kraken_k3"])] for r in r0])
print()
print("=== S3 frame_metrics（非周期） ===")
print("  thermocline displacement: min %.3f max %.3f  (span %.3f m)" % (min(eta2), max(eta2), max(eta2) - min(eta2)))
print("  truth k drift (max-min per mode):", np.array2string(kk.max(axis=0) - kk.min(axis=0), precision=6))
print()
print("=== 可比性 ===")
print("  eta span  A'-1 %.3f m vs S3 %.3f m -> ratio %.2f" % (eta.max()-eta.min(), max(eta2)-min(eta2), (eta.max()-eta.min())/max(max(eta2)-min(eta2),1e-12)))
