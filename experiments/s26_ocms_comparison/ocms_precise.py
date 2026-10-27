"""Precise OCMS: coarse-to-fine search over xi1 with an l1-constrained fit.

Selection rule (in this order):
  1. the constraint ||Psi a - p||_2 <= eps must be satisfied;
  2. among feasible candidates, the number of active atoms must equal K;
  3. among those, keep the smallest ||a||_1.
"""
import sys, time
import numpy as np
import cvxpy as cp

REPO = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo"
sys.path[:0] = [REPO + r"\src", r"E:\Users\lenovo\_pdf_extract"]

from ocms_compare_v2 import ocms_dictionary
from mode_fri.scenario import ShallowWaterScenario

H, C0, FREQ, DZ = 40.0, 1500.0, 300.0, 0.125
Z = np.arange(0.0, H + DZ * 0.5, DZ)
K = 3


def solve_l1(Psi, p, eps):
    """Return (feasible, l1, n_active, k_est_idx) for one dictionary."""
    a = cp.Variable(Psi.shape[1], complex=True)
    prob = cp.Problem(cp.Minimize(cp.norm1(a)), [cp.norm(Psi @ a - p, 2) <= eps])
    try:
        prob.solve(solver=cp.CLARABEL)
    except Exception:
        return None
    if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    av = np.asarray(a.value).ravel()
    resid = float(np.linalg.norm(p - Psi @ av))
    mag = np.abs(av)
    n_active = int(np.sum(mag > 0.10 * mag.max())) if mag.max() > 0 else 0
    return dict(feasible=resid <= eps * 1.02, resid=resid, l1=float(prob.value),
                n_active=n_active, a=av)


def evaluate_xi1(xi1, sc, p, idx, eps, dxi, xi_min, max_modes=16):
    xis, Psi = ocms_dictionary(sc, xi1, dxi, xi_min, max_modes)
    if Psi.shape[1] < K:
        return None
    out = solve_l1(Psi[idx], p, eps)
    if out is None:
        return None
    out["xi1"] = float(xi1)
    out["xis"] = xis
    return out


def ocms_precise(Y, receivers, sc, coarse=160, refine_rounds=4, dxi_factor=0.004,
                 eps=None, eps_factor=1.0):
    """Coarse grid then successive local refinement around the best feasible xi1.

    eps should be the noise level ||noise||_2; OCMS is sensitive to it, so pass
    the measured value when available.  eps_factor scales a fallback estimate.
    """
    k0 = 2 * np.pi * FREQ / C0
    xi_min = 0.80 * k0
    dxi = dxi_factor * (k0 - xi_min)
    p = Y[:, 0]
    idx = np.array([int(np.argmin(np.abs(Z - r))) for r in receivers])
    if eps is None:
        eps = eps_factor * np.linalg.norm(p) * 10 ** (-10 / 20)

    t0 = time.perf_counter()
    grid = np.linspace(k0 * 0.9999, xi_min + dxi, coarse)
    cands = [evaluate_xi1(x, sc, p, idx, eps, dxi, xi_min) for x in grid]
    cands = [c for c in cands if c is not None]
    if not cands:
        return None, time.perf_counter() - t0

    def score(c):
        return (0 if c["feasible"] else 1,
                abs(c["n_active"] - K),
                c["l1"])

    cands.sort(key=score)
    best = cands[0]
    span = (k0 * 0.9999 - xi_min) / coarse

    for r in range(refine_rounds):
        span = span / 2.0
        grid = np.linspace(best["xi1"] - span, best["xi1"] + span, 21)
        newc = [evaluate_xi1(x, sc, p, idx, eps, dxi, xi_min) for x in grid]
        newc = [c for c in newc if c is not None] + [best]
        newc.sort(key=score)
        best = newc[0]

    av = best["a"]
    top = np.argsort(np.abs(av))[::-1][:K]
    k_est = np.sort(best["xis"][top])[::-1]
    elapsed = time.perf_counter() - t0
    return dict(k_est=k_est, xi1=best["xi1"], l1=best["l1"],
                resid=best["resid"], feasible=best["feasible"],
                n_active=best["n_active"], seconds=elapsed), elapsed


if __name__ == "__main__":
    from aperture_sweep import build_frame
    receivers = np.round(np.linspace(1.0, 40.0, 40) / DZ) * DZ
    Y, k_true, _ = build_frame(receivers, 20260901)
    sc = ShallowWaterScenario(water_depth=H, dz=DZ, frequency=FREQ, sound_speed=C0)
    res, _ = ocms_precise(Y, receivers, sc)
    print("true k :", np.round(k_true, 6))
    print("OCMS   :", np.round(res["k_est"], 6))
    print(f"xi1={res['xi1']:.6f} l1={res['l1']:.1f} resid={res['resid']:.2f} "
          f"active={res['n_active']} ({res['seconds']:.0f}s)")
    print("RMSE   :", np.sqrt(np.mean((res["k_est"] - k_true) ** 2)))
