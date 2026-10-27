"""S26 rerun: true Diff-FRI (dynamic_topn_refined) vs OCMS on 20 paired realizations.

Fixes vs the original ocms_csmt_compare.py:
  1. Diff-FRI now calls DYN.dynamic_topn_refined (the estimator described in Sec. 2.3),
     not direct_estimator.estimate / _train_multistage.
  2. Both initializers see the SAME 20-snapshot first frame (OCMS extended from a
     single snapshot to a multi-snapshot group-sparse program).
  3. 20 independent paired coefficient/noise realizations instead of a single run.
  4. OCMS modal count still taken as the K largest row norms (|a| rule retained).

Usage:  python s26_rerun.py --start 0 --end 1 --out <csv path>
"""
import argparse, csv, importlib.util, sys, time
from pathlib import Path

import numpy as np
import cvxpy as cp

REPO = r"E:\Users\lenovo\NNFRI\01_论文稿件\合作机\10.19\repo"
EXTRACT = r"E:\Users\lenovo\_pdf_extract"
sys.path[:0] = [REPO + r"\src", EXTRACT]

from ssp_wavenumbers import solve_all_wavenumbers
from mode_fri.mode_solver import solve_mode
from mode_fri.data_generator import generate_complex_coefficients, add_complex_white_noise
from mode_fri.causal_ssp_modal_tracking import CSMTConfig, CausalSSPModalTracking
from mode_fri.scenario import ShallowWaterScenario
from ocms_compare_v2 import ocms_dictionary

DYN_PATH = Path(REPO) / "data" / "vla_geometry" / "scripts" / "dynamic_local_basis_coupling_suite.py"
_spec = importlib.util.spec_from_file_location("dynsuite", str(DYN_PATH))
DYN = importlib.util.module_from_spec(_spec)
sys.modules["dynsuite"] = DYN
_spec.loader.exec_module(DYN)

H, C0, FREQ, DZ = 40.0, 1500.0, 300.0, 0.125
Z = np.arange(0.0, H + DZ * 0.5, DZ)
K = 3
ORDERS = (1, 2, 5)
N_FRAMES = 21
N_SNAPSHOTS = 20
RECEIVERS = np.linspace(1.0, 40.0, 40)
SNR_DB = 10.0
SUCCESS_TOL = 1.5e-3
RIDX = np.array([int(np.argmin(np.abs(Z - r))) for r in RECEIVERS])

# canonical Diff-FRI configuration (protocols/synthetic_40m.json)
CFG_FRI = dict(grid_size=512, top_n_hypotheses=5, beam_branch_factor=6,
               beam_width=24, grid_min_separation_rad_m=0.00066738, gate_tolerance=1e-12)


def dynamic_ssp(t_frac, amplitude_m=0.35, center=20.0, sigma=8.0):
    eta = amplitude_m * np.sin(2 * np.pi * t_frac)
    taper = np.exp(-0.5 * ((Z - center) / sigma) ** 2)
    shifted = np.clip(Z - eta * taper, Z[0], Z[-1])
    return np.interp(shifted, Z, np.full_like(Z, C0)), eta


def build_frame_bases():
    """The SSP trajectory and its reference wavenumbers are deterministic: precompute once.

    Only the modal coefficients and the noise vary across realizations, so rebuilding
    the KRAKEN dictionaries per realization is redundant work.
    """
    bases = []
    for t in range(N_FRAMES):
        ssp_t, _ = dynamic_ssp(t / (N_FRAMES - 1))
        k_true = solve_all_wavenumbers(ssp_t, FREQ, DZ, max_modes=ORDERS[-1] + 1)
        k_use = np.array([k_true[o - 1] for o in ORDERS])
        omega = 2 * np.pi * FREQ
        Phi = np.column_stack([solve_mode(float(k), omega, ssp_t, DZ) for k in k_use])
        bases.append((ssp_t, k_use, Phi))
    return bases


def realize_frames(bases, seed):
    """One paired realization: fresh coefficients and noise on the frozen trajectory."""
    frames, cleans = [], []
    for (ssp_t, k_use, Phi) in bases:
        Cmat = generate_complex_coefficients(K, N_SNAPSHOTS, random_seed=seed)
        clean = Phi @ Cmat
        noisy, _ = add_complex_white_noise(clean, snr_db=SNR_DB, random_seed=seed + 5000)
        frames.append((noisy[RIDX], k_use)); cleans.append(clean[RIDX])
        seed += 1
    return frames, cleans


def ocms_estimate_multi(Y, ssp_t, eps_l, n_grid=160, dxi_factor=0.004, max_modes=16):
    """OCMS extended to all L snapshots: group-sparse rows, per-snapshot residual caps."""
    sc = ShallowWaterScenario(water_depth=H, dz=DZ, frequency=FREQ, sound_speed=C0)
    k0 = 2 * np.pi * FREQ / ssp_t.min()
    xi_min = 0.80 * k0
    dxi = dxi_factor * (k0 - xi_min)
    L = int(Y.shape[1])

    def score_at(xi1):
        xis, Psi = ocms_dictionary(sc, xi1, dxi, xi_min, max_modes, ssp=ssp_t)
        if Psi.shape[1] < K:
            return None
        Psi = Psi[RIDX]
        M = Psi.shape[1]
        A = cp.Variable((M, L), complex=True)
        cons = [cp.norm(Psi @ A[:, l] - Y[:, l], 2) <= float(eps_l[l]) for l in range(L)]
        obj = cp.Minimize(cp.sum(cp.vstack([cp.norm(A[m, :], 2) for m in range(M)])))
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.CLARABEL)
        except Exception:
            return None
        if A.value is None or prob.status not in ("optimal", "optimal_inaccurate"):
            return None
        Av = np.asarray(A.value)
        mag = np.linalg.norm(Av, axis=1)
        n_act = int(np.sum(mag > 0.10 * mag.max())) if mag.max() > 0 else 0
        top = np.argsort(mag)[::-1][:K]
        return dict(xi1=float(xi1), n_act=n_act, l1=float(prob.value),
                    xis=xis, k_est=np.sort(xis[top])[::-1])

    cands = []
    for xi1 in np.linspace(k0 * 0.9999, xi_min + dxi, n_grid):
        r = score_at(xi1)
        if r is not None:
            cands.append(r)
    if not cands:
        return None
    # Order purely by the group-sparse objective. The l2,1 minimum is smooth and
    # single-basin around the true xi1, whereas the active-atom count n_act jumps
    # from 2 to 3 partway across the basin: filtering on n_act == K (as the
    # single-snapshot version did) rejects the basin interior and selects a
    # higher-objective xi1 whose top-K atoms are the wrong modes. n_act is still
    # reported, it is just not used to select.
    cands.sort(key=lambda c: c["l1"])
    best = cands[0]
    span = (k0 * 0.9999 - xi_min) / n_grid
    for _ in range(5):
        span /= 2.0
        for x in np.linspace(best["xi1"] - span, best["xi1"] + span, 21):
            r = score_at(x)
            if r is None:
                continue
            if r["l1"] < best["l1"]:
                best = r
    return best["k_est"]


def run_chain(init_k, frames, ssp_list):
    cfg = CSMTConfig()
    tr = CausalSSPModalTracking(cfg)
    Y0, _ = frames[0]
    tr.initialize_from_full_estimate(
        k_hat=np.asarray(init_k, float), SSP_t=ssp_list[0], Y_t=Y0,
        frequency_hz=FREQ, dz=DZ, receiver_depths=RECEIVERS,
        water_depth=H, full_runtime_seconds=0.0)
    est = [np.asarray(init_k, float)]
    for t in range(1, N_FRAMES):
        Yt, _ = frames[t]
        res = tr.step(Y_t=Yt, SSP_t=ssp_list[t], frequency_hz=FREQ, dz=DZ,
                      receiver_depths=RECEIVERS, water_depth=H)
        est.append(np.asarray(res.k_hat, float))
    truth = np.array([f[1] for f in frames])
    track_rmse = float(np.sqrt(np.mean((np.array(est[1:]) - truth[1:]) ** 2)))
    track_ok = float(np.mean(np.all(np.abs(np.array(est[1:]) - truth[1:]) <= SUCCESS_TOL, axis=1)))
    return track_rmse, track_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=20)
    ap.add_argument("--out", default=r"E:\Users\lenovo\_dsh_work\s26_rerun.csv")
    a = ap.parse_args()

    BASES = build_frame_bases()
    rows = []
    for r in range(a.start, a.end):
        t_real = time.perf_counter()
        coef_seed = 20260801 + 1000 * r
        frames, cleans = realize_frames(BASES, coef_seed)
        ssps = [b[0] for b in BASES]
        Y0, k_true0 = frames[0][0], frames[0][1]

        meta = DYN.DynamicMetadata(K=K, frequency=FREQ, array_geometry=RECEIVERS.copy(),
                                   water_depth=H, SSP=ssps[0].copy())
        t0 = time.perf_counter()
        try:
            k_fri, _, acc_fri, _, _ = DYN.dynamic_topn_refined(Y0, meta, 20260917 + 1000 * r, CFG_FRI, DZ)
            k_fri = np.sort(np.asarray(k_fri, float))[::-1]
        except Exception as e:
            print(f"[r{r}] Diff-FRI FAILED: {e}", flush=True); k_fri = None; acc_fri = False
        t_fri = time.perf_counter() - t0

        eps_l = [float(np.linalg.norm(Y0[:, l] - cleans[0][:, l])) for l in range(N_SNAPSHOTS)]
        t0 = time.perf_counter()
        k_ocms = ocms_estimate_multi(Y0, ssps[0], eps_l)
        t_ocms = time.perf_counter() - t0

        def summary(k):
            if k is None:
                return (float("nan"), 0.0)
            k = np.asarray(k, float)
            if k.size != K:
                return (float("nan"), 0.0)
            rmse = float(np.sqrt(np.mean((k - k_true0) ** 2)))
            ok = float(np.all(np.abs(k - k_true0) <= SUCCESS_TOL))
            return rmse, ok

        oi_rmse, oi_ok = summary(k_ocms)
        fi_rmse, fi_ok = summary(k_fri)
        o_tr, o_tr_ok = run_chain(k_ocms, frames, ssps) if k_ocms is not None else (float("nan"), 0.0)
        f_tr, f_tr_ok = run_chain(k_fri, frames, ssps) if k_fri is not None else (float("nan"), 0.0)

        row = dict(realization=r, coef_seed=coef_seed, fri_seed=20260917 + 1000 * r,
                   ocms_init_rmse=oi_rmse, fri_init_rmse=fi_rmse,
                   ocms_init_success=oi_ok, fri_init_success=fi_ok,
                   ocms_track_rmse=o_tr, fri_track_rmse=f_tr,
                   ocms_track_success=o_tr_ok, fri_track_success=f_tr_ok,
                   fri_gate_accepted=int(bool(acc_fri)),
                   ocms_init_seconds=t_ocms, fri_init_seconds=t_fri,
                   realization_seconds=time.perf_counter() - t_real)
        rows.append(row)
        print(f"[r{r}] OCMS init={oi_rmse:.4e} ok={oi_ok:.0f} ({t_ocms:.1f}s) | "
              f"FRI init={fi_rmse:.4e} ok={fi_ok:.0f} ({t_fri:.1f}s) | "
              f"track OCMS={o_tr:.4e} FRI={f_tr:.4e} | total={row['realization_seconds']:.1f}s", flush=True)

    with open(a.out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", a.out, "rows", len(rows), flush=True)


if __name__ == "__main__":
    main()
