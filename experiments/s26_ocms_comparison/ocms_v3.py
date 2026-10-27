"""OCMS with (1) cvxpy l1 throughout, (2) adaptive noise level, (3) Newton
refinement of xi1.

Key idea: instead of assuming eps, pick eps as the value at which the l1
solution transitions to exactly K active atoms (the sparsity boundary).  Then
locate the xi1 that minimises ||a||_1 at that eps.
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


def l1_solve(Psi, p, eps, solver=cp.CLARABEL):
    a = cp.Variable(Psi.shape[1], complex=True)
    prob = cp.Problem(cp.Minimize(cp.norm1(a)), [cp.norm(Psi @ a - p, 2) <= eps])
    try:
        prob.solve(solver=solver)
    except Exception:
        return None
    if a.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
        return None
    av = np.asarray(a.value).ravel()
    resid = float(np.linalg.norm(p - Psi @ av))
    mag = np.abs(av)
    thr = 0.10 * mag.max() if mag.max() > 0 else 0.0
    return dict(a=av, resid=resid, l1=float(prob.value),
                n_active=int(np.sum(mag > thr)))


def adaptive_eps(Psi, p, k=K, n_scan=60):
    """Scan eps downward; among candidates giving exactly K active atoms keep
    the one with the smallest residual (the tightest feasible constraint)."""
    hi = float(np.linalg.norm(p))
    best = None
    for eps in np.linspace(hi, 0.01 * hi, n_scan):
        r = l1_solve(Psi, p, eps)
        if r is None or r["n_active"] != k:
            continue
        if best is None or r["resid"] < best[1]:
            best = (float(eps), r["resid"])
    if best is None:
        return None
    return best[0]




def dictionary_at(xi1, sc, p_idx, dxi, xi_min, max_modes=16):
    xis, Psi = ocms_dictionary(sc, xi1, dxi, xi_min, max_modes)
    if Psi.shape[1] < K:
        return None
    return xis, Psi[p_idx]


def objective(xi1, sc, p, idx, eps, dxi, xi_min):
    d = dictionary_at(xi1, sc, idx, dxi, xi_min)
    if d is None:
        return np.inf
    xis, Psi = d
    r = l1_solve(Psi, p, eps)
    if r is None:
        return np.inf
    return r["l1"]


def newton_refine(x0, sc, p, idx, eps, dxi, xi_min, h=2e-5, iters=8):
    """Finite-difference Newton on f(xi1) = ||a||_1 (minimisation)."""
    x = float(x0)
    fx = objective(x, sc, p, idx, eps, dxi, xi_min)
    for _ in range(iters):
        fp = objective(x + h, sc, p, idx, eps, dxi, xi_min)
        fm = objective(x - h, sc, p, idx, eps, dxi, xi_min)
        if not np.isfinite(fp) or not np.isfinite(fm):
            break
        g = (fp - fm) / (2 * h)
        d2 = (fp - 2 * fx + fm) / (h * h)
        if d2 <= 0:
            break
        step = -g / d2
        step = float(np.clip(step, -1e-3, 1e-3))
        x_new = x + step
        f_new = objective(x_new, sc, p, idx, eps, dxi, xi_min)
        if f_new < fx:
            x, fx = x_new, f_new
        else:
            h *= 0.5
            if h < 1e-7:
                break
    return x, fx


def ocms_v3(Y, receivers, sc, coarse=160, refine_rounds=3, dxi_factor=0.004):
    k0 = 2 * np.pi * FREQ / C0
    xi_min = 0.80 * k0
    dxi = dxi_factor * (k0 - xi_min)
    p = Y[:, 0]
    idx = np.array([int(np.argmin(np.abs(Z - r))) for r in receivers])
    t0 = time.perf_counter()

    # 1) coarse grid, adaptive eps per candidate
    grid = np.linspace(k0 * 0.9999, xi_min + dxi, coarse)
    cands = []
    for xi1 in grid:
        d = dictionary_at(xi1, sc, idx, dxi, xi_min)
        if d is None:
            continue
        xis, Psi = d
        eps = adaptive_eps(Psi, p)
        r = l1_solve(Psi, p, eps)
        if r is None or r["n_active"] != K:
            continue
        cands.append((r["l1"], float(xi1), eps, r))
    if not cands:
        return None, time.perf_counter() - t0

    cands.sort(key=lambda c: c[0])
    l1_best, xi_best, eps_best, r_best = cands[0]
    span = (k0 * 0.9999 - xi_min) / coarse

    # 2) local grid refinement around the best
    for _ in range(refine_rounds):
        span /= 2.0
        for x in np.linspace(xi_best - span, xi_best + span, 15):
            d = dictionary_at(x, sc, idx, dxi, xi_min)
            if d is None:
                continue
            xis, Psi = d
            e = adaptive_eps(Psi, p) if abs(x - xi_best) > span * 0.5 else eps_best
            r = l1_solve(Psi, p, e)
            if r is None or r["n_active"] != K:
                continue
            if r["l1"] < l1_best:
                l1_best, xi_best, eps_best, r_best = r["l1"], float(x), e, r

    # 3) Newton polish
    xi_fin, l1_fin = newton_refine(xi_best, sc, p, idx, eps_best, dxi, xi_min)
    d = dictionary_at(xi_fin, sc, idx, dxi, xi_min)
    if d is not None:
        xis, Psi = d
        r = l1_solve(Psi, p, adaptive_eps(Psi, p))
        if r is not None and r["n_active"] == K and r["l1"] <= l1_best * 1.02:
            r_best, xi_best = r, xi_fin

    top = np.argsort(np.abs(r_best["a"]))[::-1][:K]
    k_est = np.sort(dictionary_at(xi_best, sc, idx, dxi, xi_min)[0][top])[::-1]
    return dict(k_est=k_est, xi1=xi_best, eps=eps_best, l1=r_best["l1"],
                n_active=r_best["n_active"],
                seconds=time.perf_counter() - t0), time.perf_counter() - t0


if __name__ == "__main__":
    from ssp_wavenumbers import solve_all_wavenumbers
    from mode_fri.mode_solver import solve_mode
    from mode_fri.data_generator import generate_complex_coefficients, add_complex_white_noise

    receivers = np.round(np.linspace(1.0, 40.0, 40) / DZ) * DZ
    ssp = np.full_like(Z, C0)
    k_all = solve_all_wavenumbers(ssp, FREQ, DZ, max_modes=6)
    k_true = np.array([k_all[o - 1] for o in (1, 2, 5)])
    omega = 2 * np.pi * FREQ
    Phi = np.column_stack([solve_mode(float(k), omega, ssp, DZ) for k in k_true])
    idx = np.array([int(np.argmin(np.abs(Z - r))) for r in receivers])
    sc = ShallowWaterScenario(water_depth=H, dz=DZ, frequency=FREQ, sound_speed=C0)

    for seed in [20260901, 20261301]:
        Cmat = generate_complex_coefficients(K, 20, random_seed=seed)
        noisy, _ = add_complex_white_noise(Phi @ Cmat, snr_db=10.0, random_seed=seed + 5000)
        Y = noisy[idx]
        res, _ = ocms_v3(Y, receivers, sc)
        if res is None:
            print(f"seed {seed}: FAILED"); continue
        rmse = np.sqrt(np.mean((res["k_est"] - k_true) ** 2))
        print(f"seed {seed}: k={np.round(res['k_est'],6)} xi1={res['xi1']:.6f} "
              f"eps={res['eps']:.1f} nact={res['n_active']} RMSE={rmse:.3e} ({res['seconds']:.0f}s)")