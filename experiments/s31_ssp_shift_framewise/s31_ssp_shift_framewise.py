"""S31: framewise-estimation control under a uniform +-1 m/s inverse-SSP offset.

Question
--------
S25 reports that SMT keeps a success fraction of 0.9875 when the inverse SSP is
offset by +-1 m/s. Does SMT still beat independent framewise estimation under
the same offset, or does the margin vanish?

Design
------
Same fixed KRAKEN-like forward construction as S25: truth SSP trajectory,
21 frames, 20 paired coefficient/noise realisations, 10 dB, 20 snapshots.
Observations are synthesised with the truth SSP; the estimator receives the
offset profile ssp +/- 1 m/s. Both arms see the identical observations.

  smt      : one full initialization at frame 0 (offset profile), then the
             SMT update with the offset profile.
  framewise: one full initialization per frame with the offset profile.

Both arms use the same initializer: beam search over a 512-point grid
(branch 6, beam 24, five candidates) followed by the acceptance-tested
continuous refinement of direct_estimator.estimate_from_coarse.

Usage
-----
python s31_ssp_shift_framewise.py --delta -1 --repeats 0 20
Outputs results/s31_delta{-1,+1}.csv
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from mode_fri.direct_estimator import estimate_from_coarse  # noqa: E402
from mode_fri.mode_solver import solve_mode  # noqa: E402

CONFIG = REPO / "configs" / "dynamic_local_basis_coupling_suite.json"
DYN_PATH = REPO / "data" / "vla_geometry" / "scripts" / "dynamic_local_basis_coupling_suite.py"


def load_dyn():
    spec = importlib.util.spec_from_file_location("dyn", DYN_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["dyn"] = module
    spec.loader.exec_module(module)
    return module


def canonicalize(column):
    x = np.asarray(column, dtype=np.complex128).reshape(-1)
    x = x / float(np.linalg.norm(x))
    pivot = int(np.argmax(np.abs(x)))
    x = x * np.exp(-1j * np.angle(x[pivot]))
    if np.real(x[pivot]) < 0.0:
        x = -x
    return x


def build_basis(wavenumbers, omega, ssp, dz, indices):
    return np.column_stack(
        [canonicalize(solve_mode(float(k), omega, ssp, dz)[indices]) for k in wavenumbers]
    ).astype(np.complex128)


def projection_scores(Y, basis):
    coeff, *_ = np.linalg.lstsq(basis, Y, rcond=None)
    residual = Y - basis @ coeff
    sig = np.sum(np.abs(Y) ** 2, axis=0)
    err = np.sum(np.abs(residual) ** 2, axis=0)
    q = err / np.maximum(sig, 1e-30)
    return float(np.mean(q)), float(np.max(q))


def random_coefficients(K, snapshots, seed):
    rng = np.random.default_rng(int(seed))
    row_scale = np.exp(-0.22 * np.arange(int(K), dtype=np.float64))
    coeff = (
        rng.standard_normal((int(K), int(snapshots)))
        + 1j * rng.standard_normal((int(K), int(snapshots)))
    ) / math.sqrt(2.0)
    return (coeff * row_scale[:, None]).astype(np.complex128)


def add_noise(data, snr_db, seed):
    clean = np.asarray(data, dtype=np.complex128)
    rng = np.random.default_rng(int(seed))
    noise = (
        rng.standard_normal(clean.shape) + 1j * rng.standard_normal(clean.shape)
    ) / math.sqrt(2.0)
    signal = float(np.linalg.norm(clean) ** 2)
    noise_power = float(np.linalg.norm(noise) ** 2)
    scale = math.sqrt(signal / max(noise_power * 10.0 ** (snr_db / 10.0), 1.0e-30))
    return (clean + scale * noise).astype(np.complex128)


def full_estimate(Y, cfg, dyn, ssp_inv, seed):
    """Beam search + acceptance-tested continuous refinement."""
    metadata = dyn.DynamicMetadata(
        K=3,
        frequency=float(cfg["frequency_hz"]),
        array_geometry=np.asarray(RECEIVERS),
        water_depth=float(cfg["water_depth_m"]),
        SSP=np.asarray(ssp_inv, dtype=np.float64),
    )
    hypotheses = dyn.dynamic_beam_hypotheses(
        Y,
        metadata,
        grid_size=512,
        top_n=5,
        branch_factor=6,
        beam_width=24,
        minimum_separation=float(cfg["estimation"]["grid_min_separation_rad_m"]),
        dz=float(cfg["dz_m"]),
    )
    scored = []
    for hyp in hypotheses:
        mean_q, worst_q = projection_scores(
            Y, build_basis(hyp, OMEGA, ssp_inv, DZ, INDICES)
        )
        scored.append((worst_q, mean_q, hyp))
    scored.sort(key=lambda item: (item[0], item[1]))
    best = scored[0][2]
    return estimate_from_coarse(
        Y,
        K=3,
        frequency=float(cfg["frequency_hz"]),
        array_geometry=np.asarray(RECEIVERS),
        water_depth=float(cfg["water_depth_m"]),
        SSP=np.asarray(ssp_inv, dtype=np.float64),
        seed=int(seed),
        initial_wavenumbers=np.asarray(best, dtype=np.float64),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delta", type=float, required=True)
    parser.add_argument("--repeats", type=int, nargs=2, default=[0, 20])
    args = parser.parse_args()

    dyn = load_dyn()
    cfg = json.loads(CONFIG.read_text())
    global OMEGA, DZ, INDICES, RECEIVERS
    OMEGA = 2.0 * math.pi * float(cfg["frequency_hz"])
    DZ = float(cfg["dz_m"])
    array = cfg["receiver_array"]
    RECEIVERS = np.arange(
        array["start_depth_m"],
        array["end_depth_m"] + 0.5 * array["spacing_m"],
        array["spacing_m"],
    )
    INDICES = np.rint(RECEIVERS / DZ).astype(np.int64)

    env = np.load(Path(__file__).resolve().parent / "env.npz")
    ssp, kraken = env["ssp"], env["kraken"]
    n_time = ssp.shape[0]

    out_dir = Path(
        os.environ.get("S31_OUT", str(Path(__file__).resolve().parent / "results"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for r in range(args.repeats[0], args.repeats[1]):
        # observations from the truth profile
        obs = []
        for t in range(n_time):
            truth_basis = build_basis(kraken[r, t], OMEGA, ssp[t], DZ, INDICES)
            clean = truth_basis @ random_coefficients(3, 20, 7000 + 100 * r + t)
            obs.append(add_noise(clean, 10.0, 9000 + 100 * r + t))

        ssp_inv = [s + args.delta for s in ssp]
        # SMT: one initialization at frame 0, then updates
        k0 = full_estimate(obs[0], cfg, dyn, ssp_inv[0], 61000 + r)
        from mode_fri.causal_ssp_modal_tracking import CSMTConfig, CausalSSPModalTracking

        tracker = CausalSSPModalTracking(CSMTConfig(local_coordinate_sweeps=2))
        tracker.initialize_from_full_estimate(
            k_hat=k0,
            SSP_t=ssp_inv[0],
            Y_t=obs[0],
            frequency_hz=float(cfg["frequency_hz"]),
            dz=DZ,
            receiver_depths=np.asarray(RECEIVERS),
            water_depth=float(cfg["water_depth_m"]),
            full_runtime_seconds=0.0,
        )
        smt_k = [np.asarray(k0, dtype=np.float64)]
        for t in range(1, n_time):
            res = tracker.step(
                Y_t=obs[t],
                SSP_t=ssp_inv[t],
                frequency_hz=float(cfg["frequency_hz"]),
                dz=DZ,
                receiver_depths=np.asarray(RECEIVERS),
                water_depth=float(cfg["water_depth_m"]),
            )
            smt_k.append(res.k_hat)

        # framewise: one initialization per frame
        fw_k = [
            full_estimate(obs[t], cfg, dyn, ssp_inv[t], 81000 + 100 * r + t)
            for t in range(n_time)
        ]

        smt_k = np.asarray(smt_k)
        fw_k = np.asarray(fw_k)
        e_smt = (smt_k - kraken[r]) ** 2
        e_fw = (fw_k - kraken[r]) ** 2
        rows.append(
            {
                "delta_m_s": args.delta,
                "repeat": r,
                "smt_rmse": float(np.sqrt(e_smt.mean())),
                "framewise_rmse": float(np.sqrt(e_fw.mean())),
                "smt_success": float(np.mean(np.max(np.abs(smt_k - kraken[r]), axis=1) < 1.5e-3)),
                "framewise_success": float(np.mean(np.max(np.abs(fw_k - kraken[r]), axis=1) < 1.5e-3)),
            }
        )
        print(
            "repeat %2d  delta %+.1f  smt %.3e  fw %.3e"
            % (r, args.delta, rows[-1]["smt_rmse"], rows[-1]["framewise_rmse"]),
            flush=True,
        )

    out = out_dir / (
        "s31_delta%+d_r%d-%d.csv" % (int(args.delta), args.repeats[0], args.repeats[1])
    )
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
