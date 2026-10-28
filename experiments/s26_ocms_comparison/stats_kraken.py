import csv, statistics as st, sys, numpy as np
sys.stdout.reconfigure(encoding="utf-8")
import os
R = os.path.dirname(os.path.abspath(__file__))
if len(sys.argv) > 1:
    CSV = sys.argv[1]
else:
    CSV = os.path.join(R, "s27_precheck_kraken.csv")

def load(p):
    return list(csv.DictReader(open(p, encoding="utf-8-sig")))

rows = load(CSV)
g = lambda k: [float(r[k]) for r in rows]
print("=== A'-1 精修消融（KRAKEN 基, n=%d） ===" % len(rows))
for k in ["init_coarse", "init_refined", "init_forced", "track_coarse", "track_refined", "track_forced"]:
    v = g(k); print("  %-14s median %.3e  mean %.3e" % (k, st.median(v), st.mean(v)))
gate = [int(r["gate_accepted"]) for r in rows]
print("  gate accepted: %d/%d" % (sum(gate), len(gate)))
# 配对差
ic, ir, iff = g("init_coarse"), g("init_refined"), g("init_forced")
tc, tr, tf = g("track_coarse"), g("track_refined"), g("track_forced")
print()
print("=== 强制精修 vs 粗搜 ===")
print("  init  中位相对变化: %+.2f%%  (worst %+.1f%%)" % (
    100*st.median([(b-a)/a for a,b in zip(ic,iff)]), 100*max((b-a)/a for a,b in zip(ic,iff))))
print("  track 中位相对变化: %+.4f%%  (worst %+.2f%%)" % (
    100*st.median([(b-a)/a for a,b in zip(tc,tf)]), 100*max((b-a)/a for a,b in zip(tc,tf))))
d = [b-a for a,b in zip(tc,tf)]
print("  track 配对差中位: %+.3e  最大绝对值 %.3e  |  <1e-6 的比例 %d/%d" % (
    st.median(d), max(abs(x) for x in d), sum(1 for x in d if abs(x) < 1e-6), len(d)))
print()
print("=== 门控精修 vs 粗搜 ===")
d2 = [b-a for a,b in zip(tc,tr)]
print("  track 配对差中位: %+.3e  最大绝对值 %.3e" % (st.median(d2), max(abs(x) for x in d2)))
