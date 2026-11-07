"""S32: SAVEX15 framewise control at 40 m and 101 m.

For each geometry, rebuild the same observations used by the SAVEX15 Stage-B
archive (same SHA256 seed schedule, same 10 dB), take 21 evenly spaced frames
(every seventh) and 10 of the 20 realisations, and run the full initializer
(beam 512 + acceptance-tested refinement) on every frame.  The SMT arm is read
from the archived Stage-B CSV, so both arms share the identical observation.

Usage
-----
python s32_savex_framewise.py --geom design_40 --reps 0 10
Outputs results/s32_{geom}.csv
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
DATA = REPO / "data" / "savex15"

from mode_fri.direct_estimator import estimate_from_coarse  # noqa: E402
from mode_fri.mode_solver import solve_mode  # noqa: E402

FREQ = 300.0
C_NOMINAL = 1500.0
START_SEED = 20150520
N_SNAPSHOTS = 20
SNR_DB = 10.0
MODE_AMPLITUDE_POWER = 1.5
FULL = {"grid_size": 512, "top_n_hypotheses": 5, "beam_branch_factor": 6,
        "beam_width": 24, "grid_min_separation_rad_m": 0.004}
GEOM = {"sav15_101": 101.0, "design_40": 40.0}
DYN_PATH = REPO / "data" / "vla_geometry" / "scripts" / "dynamic_local_basis_coupling_suite.py"


def stable_seed(*items):
    blob = "|".join(str(x) for x in items).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


def homogeneous_scale(H):
    omega = 2.0 * math.pi * FREQ
    k0 = omega / C_NOMINAL
    def gap(h):
        ks = [math.sqrt(k0 * k0 - (m * math.pi / h) ** 2) for m in (1, 2, 3)]
        return float(np.median(np.diff(ks) * -1))
    return gap(H) / gap(40.0)


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
    x = x * np.exp(-1j * np.angle(pivot))
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


def generated_y(geom, rep, t, forward):
    n = int(forward["n_modes_per_frame"][t])
    roots = np.real(forward["roots"])[t, :n]
    b = forward["receiver_mode_shapes"][t, :, :n]
    b = b / np.linalg.norm(b, axis=0, keepdims=True)
    rng_c = np.random.default_rng(stable_seed("NN_FRI_Bv1_C", geom, START_SEED + rep, t))
    coef = (rng_c.normal(size=(n, N_SNAPSHOTS)) + 1j * rng_c.normal(size=(n, N_SNAPSHOTS))) / math.sqrt(2)
    coef *= (np.arange(1, n + 1, dtype=float) ** (-MODE_AMPLITUDE_POWER))[:, None]
    clean = b @ coef
    rng_n = np.random.default_rng(stable_seed("NN_FRI_Bv1_N", geom, START_SEED + rep, t))
    noise = (rng_n.normal(size=clean.shape) + 1j * rng_n.normal(size=clean.shape)) / math.sqrt(2)
    scale = float(np.linalg.norm(clean) / max(np.linalg.norm(noise), 1e-30) / math.sqrt(10 ** (SNR_DB / 10)))
    return clean + noise * scale, roots[:3]


def full_estimate(Y, ssp, geom, seed, dyn):
    dz = GEOM[geom] / (ssp.size - 1)
    receivers = np.load(DATA / "arrays" / ("SSP_INPUT_%s.npz" % geom))["receiver_depths_m"]
    indices = np.rint(receivers / dz).astype(np.int64)
    metadata = dyn.DynamicMetadata(
        K=3, frequency=FREQ, array_geometry=np.asarray(receivers),
        water_depth=GEOM[geom], SSP=np.asarray(ssp, dtype=np.float64),
    )
    fac = homogeneous_scale(GEOM[geom])
    hyps = dyn.dynamic_beam_hypotheses(
        Y, metadata,
        grid_size=FULL["grid_size"], top_n=FULL["top_n_hypotheses"],
        branch_factor=FULL["beam_branch_factor"], beam_width=FULL["beam_width"],
        minimum_separation=FULL["grid_min_separation_rad_m"] * fac, dz=dz,
    )
    scored = []
    for hyp in hyps:
        mean_q, worst_q = projection_scores(Y, build_basis(hyp, 2 * math.pi * FREQ, ssp, dz, indices))
        scored.append((worst_q, mean_q, hyp))
    scored.sort(key=lambda item: (item[0], item[1]))
    return estimate_from_coarse(
        Y, K=3, frequency=FREQ, array_geometry=np.asarray(receivers),
        water_depth=GEOM[geom], SSP=np.asarray(ssp, dtype=np.float64),
        seed=int(seed), initial_wavenumbers=np.asarray(scored[0][2], dtype=np.float64),
    )


def load_smt(geom):
    """Archived SMT (dynamic SSP) wavenumbers keyed by (trial, frame)."""
    out = {}
    for path in sorted((DATA / "results").glob("%s_rep*.csv" % geom)):
        with open(path, newline="", encoding="utf-8-sig") as h:
            for r in csv.DictReader(h):
                if r["method"] != "CSMT_dynamic_SSP":
                    continue
                out[(int(r["trial"]), int(r["frame"]))] = np.array(
                    [float(r["k1"]), float(r["k2"]), float(r["k3"])]
                )
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geom", required=True, choices=list(GEOM))
    ap.add_argument("--reps", type=int, nargs=2, default=[0, 10])
    ap.add_argument("--stride", type=int, default=7)
    args = ap.parse_args()

    dyn = load_dyn()
    forward = np.load(DATA / "arrays" / ("KRAKEN_INDEPENDENT_FORWARD_%s.npz" % args.geom), allow_pickle=True)
    ssp_in = np.load(DATA / "arrays" / ("SSP_INPUT_%s.npz" % args.geom), allow_pickle=True)
    smt = load_smt(args.geom)
    frames = list(range(0, forward["roots"].shape[0], args.stride))

    rows = []
    for rep in range(args.reps[0], args.reps[1]):
        for t in frames:
            Y, truth = generated_y(args.geom, rep, t, forward)
            k_full = full_estimate(Y, ssp_in["c"][t], args.geom, START_SEED + rep, dyn)
            k_smt = smt[(rep, t)]
            e_full = float(np.sqrt(np.mean((k_full - truth) ** 2)))
            e_smt = float(np.sqrt(np.mean((k_smt - truth) ** 2)))
            rows.append({
                "geom": args.geom, "rep": rep, "frame": t,
                "framewise_rmse": e_full, "smt_rmse": e_smt,
                "framewise_max_err": float(np.max(np.abs(k_full - truth))),
                "smt_max_err": float(np.max(np.abs(k_smt - truth))),
            })
        print("rep %2d done (last frame %d): fw %.3e smt %.3e"
              % (rep, rows[-1]["frame"], rows[-1]["framewise_rmse"], rows[-1]["smt_rmse"]), flush=True)

    out_dir = Path(os.environ.get("S32_OUT", str(Path(__file__).resolve().parent / "results")))
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / ("s32_%s_r%d-%d.csv" % (args.geom, args.reps[0], args.reps[1]))
    with open(out, "w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
