"""OCMS-like (modal-orthogonality criterion) vs Diff-FRI single-frame comparison.

Both estimators see the same single observation frame and the same SSP.
The OCMS-like estimator ranks candidate wavenumber sets by the deviation of
the trial-mode Gram matrix from the identity; Diff-FRI ranks them by the
projection residual and then refines continuously.
"""
import sys, time, csv
import numpy as np

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


def normalized_mode(sc, k, c_profile=None):
    z = sc.z
    if c_profile is None:
        c_profile = sc.sound_speed_profile()
    phi = solve_mode(horizontal_wavenumber=float(k), omega=sc.omega,
                     sound_speed_profile=c_profile, dz=sc.dz)
    n = np.sqrt(np.trapezoid(phi * phi, z))
    return phi / max(n, 1e-30)


def gram_deviation_from_rows(rows, z):
    """rows: list of normalized mode vectors."""
    Phi = np.column_stack(rows)
    G = np.trapezoid(Phi[:, :, None] * Phi[:, None, :], z, axis=0)
    return float(np.linalg.norm(G - np.eye(len(rows)), "fro"))


def ocms_like_estimate(Y, sc, grid=48, c_profile=None):
    """Rank candidate triples by modal-orthogonality deviation only."""
    z = sc.z
    k0 = sc.omega / sc.sound_speed
    kmin, kmax = 0.05 * k0, 0.999 * k0
    grid_k = np.linspace(kmin, kmax, grid)

    pre = {}
    for k in grid_k:
        pre[k] = normalized_mode(sc, k, c_profile)

    best = None
    keys = list(grid_k)
    t0 = time.perf_counter()
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            for m in range(j + 1, len(keys)):
                dev = gram_deviation_from_rows([pre[keys[i]], pre[keys[j]], pre[keys[m]]], z)
                if best is None or dev < best[0]:
                    best = (dev, np.array([keys[i], keys[j], keys[m]]))
    elapsed = time.perf_counter() - t0
    return np.sort(best[1])[::-1], elapsed, best[0]


def main():
    sc = make_scenario()
    k_true = analytic_k(sc)
    print("k_true:", k_true, flush=True)

    rows = []
    for snr in [10.0]:
        for seed in range(3):
            ds = generate_vertical_array_dataset(
                sc, k_true, number_of_snapshots=20, snr_db=snr,
                coefficient_seed=20260731 + seed, noise_seed=20260732 + seed)
            Y = ds.noisy_data
            # OCMS-like
            k_ocms, t_ocms, dev = ocms_like_estimate(Y, sc, grid=40)
            rmse_ocms = float(np.sqrt(np.mean((k_ocms - k_true) ** 2)))
            # Diff-FRI
            t0 = time.perf_counter()
            k_fri = np.sort(direct_estimator.estimate(
                Y, K, sc.frequency, sc.z, sc.water_depth, sc.sound_speed_profile(),
                seed=20260917 + seed))[::-1]
            t_fri = time.perf_counter() - t0
            rmse_fri = float(np.sqrt(np.mean((k_fri - k_true) ** 2)))
            rows.append(dict(snr=snr, seed=seed,
                             ocms_rmse=rmse_ocms, ocms_time=t_ocms, ocms_gram=dev,
                             fri_rmse=rmse_fri, fri_time=t_fri))
            print(f"snr {snr} seed {seed}: OCMS rmse {rmse_ocms:.3e} ({t_ocms:.2f}s) | "
                  f"Diff-FRI rmse {rmse_fri:.3e} ({t_fri:.2f}s)", flush=True)

    out = REPO + r"\..\Submission_Package\ocms_compare.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
