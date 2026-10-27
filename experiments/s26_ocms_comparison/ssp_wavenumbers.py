"""Solve all pressure-release modal wavenumbers for a given SSP by scanning
the bottom residual |psi(H)| and bracketing sign changes."""
import numpy as np
from scipy.optimize import brentq

import sys
REPO = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo"
sys.path[:0] = [REPO + r"\src"]
from mode_fri.mode_solver import solve_mode


def bottom_residual(k, omega, ssp, dz):
    psi = solve_mode(horizontal_wavenumber=float(k), omega=omega,
                     sound_speed_profile=ssp, dz=dz)
    return float(psi[-1])


def solve_all_wavenumbers(ssp, frequency_hz, dz, max_modes=8, n_scan=4000):
    """Return modal wavenumbers in descending order (pressure-release bottom)."""
    ssp = np.asarray(ssp, dtype=float)
    omega = 2.0 * np.pi * frequency_hz
    k_max = omega / ssp.min() * 0.99999
    k_min = omega / ssp.max() * 0.02
    grid = np.linspace(k_max, k_min, n_scan)
    vals = np.array([bottom_residual(k, omega, ssp, dz) for k in grid])

    roots = []
    for i in range(len(grid) - 1):
        a, b = vals[i], vals[i + 1]
        if a == 0.0:
            roots.append(grid[i]); continue
        if a * b < 0.0:
            ka, kb = grid[i], grid[i + 1]
            try:
                r = brentq(bottom_residual, ka, kb, args=(omega, ssp, dz),
                           xtol=1e-12, rtol=1e-12, maxiter=200)
                roots.append(float(r))
            except ValueError:
                pass
        if len(roots) >= max_modes:
            break

    roots = np.array(sorted(roots, reverse=True)[:max_modes])
    return roots


if __name__ == "__main__":
    # validation: constant sound speed, pressure-release bottom -> analytic
    H, c, freq, dz = 40.0, 1500.0, 300.0, 0.125
    z = np.arange(0.0, H + dz * 0.5, dz)
    ssp = np.full_like(z, c)
    k_num = solve_all_wavenumbers(ssp, freq, dz, max_modes=5)
    k0 = 2 * np.pi * freq / c
    k_ana = np.sqrt(k0 ** 2 - (np.arange(1, 6) * np.pi / H) ** 2)
    print("numeric:", np.round(k_num, 6))
    print("analytic:", np.round(k_ana, 6))
    if k_num.size == 5:
        print("max abs error:", np.abs(k_num - k_ana).max())
