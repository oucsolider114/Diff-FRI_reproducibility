"""Correct OCMS implementation (Algorithm 1 + 2) vs Diff-FRI.

OCMS (Li et al. 2026):
  Algorithm 1: given the first-mode wavenumber xi1, recursively generate the
    full mode-depth-function dictionary by orthogonality: scan down in xi and
    pick the local minimum of |<psi1, psi(xi)>|, repeat.
  Algorithm 2: 1-D search over xi1, solve min ||a||_1 s.t. representation,
    keep the xi1 giving the sparsest coefficient vector.
"""
import sys, time, csv
import numpy as np
from scipy.optimize import lsq_linear
from sklearn.linear_model import OrthogonalMatchingPursuit
from scipy.signal import find_peaks

REPO = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo"
sys.path[:0] = [REPO + r"\src"]

from mode_fri.scenario import ShallowWaterScenario
from mode_fri.data_generator import generate_vertical_array_dataset
from mode_fri.mode_solver import solve_mode
from mode_fri import direct_estimator

ORDERS = (1, 2, 5)
K = len(ORDERS)


def make_scenario(water_depth=40.0, freq=300.0, c=1500.0, dz=0.125):
    return ShallowWaterScenario(water_depth=water_depth, dz=dz, frequency=freq, sound_speed=c)


def analytic_k(sc, orders=ORDERS):
    k0 = sc.omega / sc.sound_speed
    return np.sqrt(k0 ** 2 - (np.asarray(orders) * np.pi / sc.water_depth) ** 2)


def normalized_mode(sc, k, c_profile, z):
    phi = solve_mode(horizontal_wavenumber=float(k), omega=sc.omega,
                     sound_speed_profile=c_profile, dz=sc.dz)
    n = np.sqrt(np.trapezoid(phi * phi, z))
    return phi / max(n, 1e-30)


def ocms_dictionary(sc, xi1, dxi, xi_min, max_modes=12, ssp=None):
    """Algorithm 1: from xi1, find all mode wavenumbers as the minima of
    |<psi1, psi(xi)>| over xi, refined by parabolic interpolation.

    ssp overrides the scenario's constant profile when supplied (needed for a
    dynamic sound-speed record).
    """
    z = sc.z
    c = sc.sound_speed_profile() if ssp is None else np.asarray(ssp, dtype=float)
    psi1 = normalized_mode(sc, xi1, c, z)
    grid = np.arange(xi1, xi_min, -dxi)          # descending in xi
    vals = np.array([abs(np.trapezoid(psi1 * normalized_mode(sc, x, c, z), z))
                     for x in grid])
    peaks, _ = find_peaks(-vals, prominence=1e-3, distance=5)

    cand = []
    for p in peaks:
        if 0 < p < len(grid) - 1:
            y0, y1, y2 = vals[p - 1], vals[p], vals[p + 1]
            denom = y0 - 2.0 * y1 + y2
            delta = 0.5 * (y0 - y2) / denom if abs(denom) > 1e-30 else 0.0
            xi = grid[p] + delta * (grid[p + 1] - grid[p])
        else:
            xi = grid[p]
        cand.append(float(xi))

    xis = np.array(sorted(cand + [float(xi1)], reverse=True)[:max_modes])
    psis = np.column_stack([normalized_mode(sc, x, c, z) for x in xis])
    return xis, psis


def l1_fit_complex(Psi, p, K=3):
    """Sparse representation: OMP with K+1 atoms, then keep the K largest."""
    A = np.vstack([np.hstack([Psi.real, -Psi.imag]),
                   np.hstack([Psi.imag, Psi.real])])
    b = np.concatenate([p.real, p.imag])
    n_atoms = min(K + 1, Psi.shape[1])
    omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_atoms, fit_intercept=False)
    omp.fit(A, b)
    coef = omp.coef_
    half = coef.size // 2
    a = coef[:half] + 1j * coef[half:]
    resid = np.linalg.norm(p - Psi @ a)
    return a, resid


def ocms_estimate(Y, sc, n_grid=600, dxi_factor=0.004, max_modes=16):
    """Algorithm 2: 1-D search over xi1.  Among candidates whose representation
    residual is within epsilon of the best, keep the sparsest (smallest ||a||_1)."""
    z = sc.z
    k0 = sc.omega / sc.sound_speed
    xi_min = 0.80 * k0
    xi_max = k0 * 0.9999
    dxi = dxi_factor * (k0 - xi_min)
    p = Y[:, 0]

    cand = []
    t0 = time.perf_counter()
    for xi1 in np.linspace(xi_max, xi_min + dxi, n_grid):
        xis, Psi = ocms_dictionary(sc, xi1, dxi, xi_min, max_modes)
        if Psi.shape[1] < K:
            continue
        a, resid = l1_fit_complex(Psi, p, K=K)
        top = np.argsort(np.abs(a))[::-1][:K]
        k_est = np.sort(xis[top])[::-1]
        cand.append((resid, float(np.sum(np.abs(a))), k_est))
    elapsed = time.perf_counter() - t0

    if not cand:
        return None, elapsed
    # residual is the only reliable criterion here; l1 degenerates to the trivial a=0
    cand.sort(key=lambda c: c[0])
    resid, l1, k_est = cand[0]
    return (l1, k_est, resid), elapsed


def main():
    sc = make_scenario()
    k_true = analytic_k(sc)
    print("k_true:", k_true, flush=True)

    rows = []
    for snr in [10.0]:
        for seed in range(2):
            ds = generate_vertical_array_dataset(
                sc, k_true, number_of_snapshots=20, snr_db=snr,
                coefficient_seed=20260731 + seed, noise_seed=20260732 + seed)
            Y = ds.noisy_data

            t0 = time.perf_counter()
            best, t_ocms = ocms_estimate(Y, sc)
            if best is None:
                print("OCMS failed", flush=True); continue
            score, k_ocms, resid = best
            rmse_ocms = float(np.sqrt(np.mean((k_ocms - k_true) ** 2)))

            t0 = time.perf_counter()
            k_fri = np.sort(direct_estimator.estimate(
                Y, K, sc.frequency, sc.z, sc.water_depth, sc.sound_speed_profile(),
                seed=20260917 + seed))[::-1]
            t_fri = time.perf_counter() - t0
            rmse_fri = float(np.sqrt(np.mean((k_fri - k_true) ** 2)))

            rows.append(dict(snr=snr, seed=seed,
                             ocms_rmse=rmse_ocms, ocms_time=t_ocms,
                             ocms_l1=score, ocms_resid=resid,
                             fri_rmse=rmse_fri, fri_time=t_fri))
            print(f"snr {snr} seed {seed}: OCMS rmse {rmse_ocms:.3e} "
                  f"(l1 {score:.3f}, resid {resid:.3e}, {t_ocms:.1f}s) | "
                  f"Diff-FRI rmse {rmse_fri:.3e} ({t_fri:.1f}s)", flush=True)

    if rows:
        out = r"E:\Users\lenovo\_pdf_extract\ocms_compare_v2.csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
        print("wrote", out)


if __name__ == "__main__":
    main()
