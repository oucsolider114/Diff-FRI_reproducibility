import csv, statistics as st, sys
sys.stdout.reconfigure(encoding="utf-8")
R = r"E:\Users\lenovo\_dsh_work"

def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))

# S3 原始
s3 = load(r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo\data\synthetic\frame_metrics.csv")
reps = sorted({int(r["repeat"]) for r in s3})
print(f"S3 frame_metrics: {len(s3)} 行, {len(reps)} realizations")
# 每次 realization 的 frame0 误差 与 tracking(1-20) 均值
init3, track3, all3 = [], [], []
for rp in reps:
    rows = sorted([r for r in s3 if int(r["repeat"]) == rp], key=lambda r: int(r["time_index"]))
    e = [float(r["csmt_k_rmse_rad_m"]) for r in rows]
    init3.append(e[0]); track3.append(sum(e[1:]) / len(e[1:])); all3.append(sum(e) / len(e))
print(f"  S3 init  (frame0) : median {st.median(init3):.3e}  mean {st.mean(init3):.3e}")
print(f"  S3 track (1-20)   : median {st.median(track3):.3e}  mean {st.mean(track3):.3e}")
print(f"  S3 all-frame      : median {st.median(all3):.3e}  mean {st.mean(all3):.3e}")

# S28 对齐后 vs 原
for tag, f in [("S28 原(6.67e-4)", "s26_precheck_dynamic.csv"), ("S28 对齐(4e-3)", "s26_precheck_aligned.csv")]:
    rows = load(f"{R}/{f}")
    io = [float(r["ocms_init_rmse"]) for r in rows]
    if_ = [float(r["fri_init_rmse"]) for r in rows]
    to = [float(r["ocms_track_rmse"]) for r in rows]
    tf = [float(r["fri_track_rmse"]) for r in rows]
    print(f"\n{tag}: n={len(rows)}")
    print(f"  init  OCMS median {st.median(io):.3e} | FRI median {st.median(if_):.3e}")
    print(f"  track OCMS median {st.median(to):.3e} | FRI median {st.median(tf):.3e}")
    d = [f - o for o, f in zip(to, tf)]
    print(f"  paired track diff (FRI-OCMS): median {st.median(d):+.3e}")
