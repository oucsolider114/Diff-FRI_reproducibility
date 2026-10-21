#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NN_FRI SAVEX15 Stage B v1: predeclared synthetic-acoustics long-horizon test.

IMPORTANT: SAVEX15 real temperature/CTD drives an INDEPENDENT KRAKEN forward
model; acoustic observations are SYNTHETIC and the seabed is assumed.
Uses unmodified frozen Full-NNFR and CSMT-v2 *source*. The geometry-normalized
variant changes only predeclared dimensional parameters, NOT source or algorithm.

Commands:
  --preflight  : verify Stage A data, source hashes, lock, branches. NO performance.
  --pilot      : 3 frames, 1 seed, first geometry; save separate PILOT results.
  --run        : predeclared 20 seeds x 144 frames x 2 geometries. Can resume
                 per completed (geometry, seed); no Stage A KRAKEN reruns.
  --summarize  : recompute and tabulate completed trials, no new computation.
  --export-min : package all 40 complete audited trials; refuses partial output.

STOP on an unverified source checksum / changed Stage A source / protocol mismatch.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import zipfile
from typing import Any

import numpy as np

# Strict checksum contract from the user's recovered original checkout.
SOURCE_SHA256 = {
    "src/mode_fri/causal_ssp_modal_tracking.py": "69a9d8d7d7da82899fad3cfce7fdc8b20d6ba3a6126d0ef42e0e8703b239497f",
    "src/mode_fri/online_track_reacquire.py": "d6acf08bd4b34337c6a953fbda33987f975fe91ab9b766f8b2377cc7de2a18d5",
    "src/mode_fri/direct_estimator.py": "8bc384df6bbeb8126f67fcc84bb7f3f2d28964dd49ce46daedd8fc041b177e0a",
    "src/mode_fri/mode_frinet.py": "ee0eb6bc544eca37d60d8e5a3f8d24901c00e0c69e67f506bd8f5dd5d0b8da05",
    "experiments/csmt_v2_independent_holdout.py": "5fcc8c460f2ba9d2acc4ab533ab58835a8841212e5a1532b077a19d756a1f500",
    "outputs/csmt_v2_independent_holdout/CSMT_V2_PARAMETER_LOCK.json": "188dfae9ff4617bf1a60427a011b0118ba23d09cd5a47dd9deff4eb6f8f66034",
    "experiments/dynamic_local_basis_coupling_suite.py": "9640c7a536afd0ffb27c604dd1c40c355181ae2f3d84d4feb6cd47338617b9a6",
    "src/mode_fri/mode_solver.py": "6a842790e66af0a1749f06bf476e9ce98ed1ff23b0399e69f18f9c2688f31773",
}

STAGE_A_SSP_SHA = {
    "sav15_101": "37a657ce7071b5b10134f40e7158cbdce7b940b819676059c481af872a7bbaa1",
    "design_40": "a68c0e7c42c965fe514ba93e1bc7ecb0b567ca65840be717d6124ebe9cdd12c4",
}
STAGE_A_SCRIPT_SHA = "a6a908451b9840d7f67f847d7e2868d1aeb4f5dff881704b0c9853daf7a5931d"
STAGE_A_FORWARD_SHA = {
    "sav15_101": "8c0803a7216f1675159ab51db1c7c36473c0642398bd800037d24e6448dccea7",
    "design_40": "ddff66d1268a2fd6fa38cb116f70f7a19a8ddf37dd80ff1459e0d3664a9031fb",
}
# Original frozen physical configuration. This lock is validated against file.
ORIG = {
    "beta_pred": 0.5, "delta0_rad_m": 0.0012,
    "delta_min_rad_m": 0.0008, "delta_max_rad_m": 0.004,
    "lambda_k": 1.5, "lambda_ssp": 0.15,
    "minimum_mode_separation_rad_m": 0.0035,
    "local_grid_points": 9, "local_coordinate_sweeps": 2,
    "enable_quadratic_subgrid": True,
}
FULL = {"grid_size": 512, "top_n_hypotheses": 5,
        "beam_branch_factor": 6, "beam_width": 24,
        "grid_min_separation_rad_m": 0.004, "gate_tolerance": 1e-12}
GEOM = {"sav15_101": 101.0, "design_40": 40.0}
FREQ = 300.0
C_NOMINAL = 1500.0  # physics-only scale anchor, not KRAKEN root or test truth
N_REPS = 20
START_SEED = 20150520
N_SNAPSHOTS = 20
SNR_DB = 10.0
# Each mode m=1..M has receiver-normalized synthetic coefficient RMS m^-1.5.
# This prior ensures the specified first three model components are normally
# observable, but is idealized and must not be claimed as field-source physics.
MODE_AMPLITUDE_POWER = 1.5
DIAG_MAC_MIN = 0.90
DIAG_MAC_MARGIN_MIN = 0.02
# Predeclared estimator-side grid (forward KRAKEN meshes remain unchanged).
# 0.5 m yields exact receiver-depth alignment for both geometries and is
# intentionally coarser than the independent KRAKEN forward solver meshes.
INVERSE_DZ_M = {"sav15_101": 0.5, "design_40": 0.5}


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(2**20), b""):
            h.update(b)
    return h.hexdigest()


def sha_array(x: np.ndarray) -> str:
    x = np.ascontiguousarray(x)
    return sha_bytes(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes())


def sha_object(x: Any) -> str:
    return sha_bytes(json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode())


def stable_seed(*items: Any) -> int:
    blob = "|".join(str(x) for x in items).encode("utf-8")
    return int.from_bytes(hashlib.sha256(blob).digest()[:4], "little")


def homogeneous_scale(H: float) -> float:
    """Initial *physics-only* geometry transfer, fixed without seeing tests."""
    def gap(h):
        omega = 2.0 * math.pi * FREQ
        k0 = omega / C_NOMINAL
        ks = [math.sqrt(k0 * k0 - (m*math.pi/h)**2) for m in (1,2,3)]
        return float(np.median(np.diff(ks) * -1))
    return gap(H) / gap(40.0)


def csmt_params(geom: str, *, unchanged_local: bool = False) -> dict:
    c = dict(ORIG)
    fac = 1.0 if unchanged_local else homogeneous_scale(GEOM[geom])
    for key in ("delta0_rad_m", "delta_min_rad_m", "delta_max_rad_m",
                "minimum_mode_separation_rad_m"):
        c[key] *= fac
    return c


def full_params(geom: str) -> dict:
    c = dict(FULL)
    c["grid_min_separation_rad_m"] *= homogeneous_scale(GEOM[geom])
    return c


def protocol() -> dict:
    return {
        "id": "SAVEX15_STAGE_B_v1_PREDECLARED",
        "data_class": "REAL T/SP observations + KRAKEN-synthesized acoustics; no SAVEX field acoustic records",
        "geometries": GEOM, "frequency_hz": FREQ,
        "seeds": [START_SEED+i for i in range(N_REPS)],
        "frames": 144, "sample_interval_s": 600,
        "snapshots_per_frame": N_SNAPSHOTS, "snr_db": SNR_DB,
        "forward": {
            "all_computed_KRAKEN_modes": True,
            "normalize_each_receiver_mode_column_l2": True,
            "modal_rms_amplitude": "m^-1.5, index m=1..M at each frame",
            "coefficient_distribution": "independent circular complex normal, each mode and snapshot",
            "noise": "complex circular normal; frame-level observed SNR exactly 10 dB",
            "random_generator": "SHA256(namespace, geom, seed, frame) => NumPy default_rng",
            "source_range_or_coherent_fixed_emitter": "NOT modeled; modal phases absorbed by free complex coefficients",
        },
        "target_model_branches": "three highest-real-k computed model roots; mode-index rank fixed",
        "branch_adjudication": {
            "test": "consecutive receiver-only full-model mode-shape MAC, rank-preserving",
            "min_diagonal_mac": DIAG_MAC_MIN, "min_diagonal_row_margin": DIAG_MAC_MARGIN_MIN,
            "fail": "mark model-truth branch ambiguity; keep acoustic metrics and report missing branch RMSE",
        },
        "inverse_water_column_grid_dz_m": INVERSE_DZ_M,
        "inverse_grid_note": "predeclared 0.5-m estimator grid, independent of Stage A forward mesh, chosen for computational tractability; no algorithm-result tuning",
        "parameter_transfer": {
            "anchor_c_m_s": C_NOMINAL,
            "scale": "median(top3 adjacent low-order homogeneous pressure-release gap @H) / same @H40",
            "applies_to": ["full grid minimum separation", "CSMT delta0, delta_min, delta_max, minimum separation"],
            "does_not_change": ["source code", "TopN/beam/gate", "beta", "lambdas", "grid points", "sweeps", "quadratic option"],
            "frozen_full_baseline": FULL, "frozen_csmt": ORIG,
            "geometry_full": {g: full_params(g) for g in GEOM},
            "geometry_csmt": {g: csmt_params(g) for g in GEOM},
        },
        "methods": ["hold_k0_fixed_SSP", "hold_k0_dynamic_SSP",
                    "CSMT_fixed_SSP", "CSMT_dynamic_SSP",
                    "unchanged_local_params_dynamic_SSP_at_101m_only"],
        "full_framewise": "not included in main trial; optional additional protocol required before claim",
        "QC_flagged_zero_based_frames": [50,95],
        "no_REACQUIRE": True,
        "no_KRAKEN_truth_as_algorithm_input": True,
        "original_dimensional_reference": "only original CSMT local parameters: uses common geometry-transferred Full init; explicitly NOT unchanged entire pipeline",
        "statistics": {
            "unit": "seed (each seed shares observations across methods)",
            "fixed_time_segments_zero_based": ["0:48", "48:96", "96:144"],
            "bootstrap_samples": 10000, "paired_bootstrap_CI": 0.95,
            "paired_bootstrap_seed_namespace": "SAVEX15_Bv1_20260930_paired_bootstrap",
            "comparisons": ["dynamic_minus_fixed", "dynamic_minus_held_dynamic", "fixed_minus_held_fixed",
                            "unchanged_local_minus_transferred_dynamic_at_101m"],
        },
        "source_hashes": SOURCE_SHA256,
        "stage_a_ssp_hashes": STAGE_A_SSP_SHA,
        "stage_a_forward_hashes": STAGE_A_FORWARD_SHA,
    }


def verify_sources(root: Path) -> dict:
    out = {}
    for rel, expect in SOURCE_SHA256.items():
        p = root / rel
        if not p.is_file():
            raise FileNotFoundError(f"Required frozen source missing: {p}")
        actual = sha_file(p)
        out[rel] = actual
        if actual != expect:
            raise RuntimeError(f"FROZEN SOURCE MISMATCH: {rel}: expected {expect}, got {actual}. STOP.")
    param = json.loads((root / "outputs/csmt_v2_independent_holdout/CSMT_V2_PARAMETER_LOCK.json").read_text(encoding="utf-8"))
    actual = param["locked_payload"]["csmt_v2"]
    for k, v in ORIG.items():
        if isinstance(v, bool):
            okay = actual[k] is v
        else:
            okay = float(actual[k]) == float(v)
        if not okay:
            raise RuntimeError(f"Frozen numeric parameter mismatch: {k} {actual[k]} != {v}")
    return out


def load_stagea(stagea: Path) -> tuple[dict, dict]:
    lock = json.loads((stagea / "EXPERIMENT_LOCK.json").read_text(encoding="utf-8"))
    if lock["code_sha256"] != STAGE_A_SCRIPT_SHA:
        raise RuntimeError("Stage A frozen source SHA in lock changed; STOP")
    if lock.get("gsw_version") != "3.6.23":
        raise RuntimeError("Stage A GSW version different from archived; STOP")
    if int(lock.get("n_time_steps", -1)) != 144 or float(lock["frequency_hz"]) != FREQ:
        raise RuntimeError("Stage A experiment schedule mismatch; STOP")
    out = {}
    for g,H in GEOM.items():
        p = stagea / f"SSP_INPUT_{g}.npz"
        if not p.is_file() or sha_file(p) != STAGE_A_SSP_SHA[g]:
            raise RuntimeError(f"Stage A SSP SHA changed for {g}; STOP")
        p2 = stagea / f"KRAKEN_INDEPENDENT_FORWARD_{g}.npz"
        if not p2.is_file():
            raise FileNotFoundError(p2)
        if sha_file(p2) != STAGE_A_FORWARD_SHA[g]:
            raise RuntimeError(f"Stage A KRAKEN forward aggregate SHA changed for {g}; STOP")
        with np.load(p2, allow_pickle=False) as f:
            data = {k: f[k] for k in f.files}
        if data["roots"].shape[0] != 144 or data["receiver_mode_shapes"].shape[0] != 144:
            raise RuntimeError(f"Stage A expected 144 frames: {g}")
        if not np.isclose(float(data["freq_hz"]), FREQ) or not np.isclose(float(data["H_m"]), H):
            raise RuntimeError(f"Stage A frequency/H mismatch: {g}")
        if data["receiver_mode_shapes"].shape[1] != data["receivers"].size:
            raise RuntimeError("Incorrect number of Stage A receiver depths")
        if not np.allclose(data["z"], np.arange(data["c"].shape[1]) * (data["z"][1]-data["z"][0]), atol=1e-8):
            raise RuntimeError(f"SSP depth grid is not uniform: {g}")
        out[g] = data
    return lock, out


def check_branches(forward: dict, g: str) -> tuple[np.ndarray, dict]:
    roots = forward["roots"]
    phi = forward["receiver_mode_shapes"][:,:,:3]
    mask = np.ones(144, dtype=bool)
    smallest_diag = 1.0
    smallest_margin = 1.0
    if np.any(~np.isfinite(np.real(roots[:,:3]))) or np.any(np.diff(np.real(roots[:,:3]),axis=1) >= 0):
        raise RuntimeError(f"First-three roots absent or not strictly descending: {g}")
    for t in range(1,144):
        a = phi[t-1]; b = phi[t]
        na = a/np.maximum(np.linalg.norm(a,axis=0,keepdims=True),1e-30)
        nb = b/np.maximum(np.linalg.norm(b,axis=0,keepdims=True),1e-30)
        m = np.abs(na.conj().T@nb)**2
        d = np.diag(m)
        margin = np.array([d[i] - np.max(np.delete(m[i,:],i)) for i in range(3)])
        smallest_diag = min(smallest_diag, float(d.min()))
        smallest_margin = min(smallest_margin, float(margin.min()))
        if d.min() < DIAG_MAC_MIN or margin.min() < DIAG_MAC_MARGIN_MIN:
            mask[t] = False
    info = {"g":g,"unambiguous_frames":int(np.count_nonzero(mask)),
            "ambiguous_frames_zero_based":np.where(~mask)[0].tolist(),
            "min_rank_diagonal_mac":smallest_diag,
            "min_row_assignment_margin":smallest_margin}
    return mask, info


def verify_lock(out: Path, p: dict, data_lock: dict, forward: dict) -> dict:
    contents = {"protocol":p,
                "stage_a_lock_sha256": sha_object(data_lock),
                "stage_a_forward_array_hashes":{
                    g:{k:sha_array(v[k]) for k in ("roots","receiver_mode_shapes","n_modes_per_frame")}
                    for g,v in forward.items()},
                "script_sha256":sha_file(Path(__file__).resolve())}
    contents["lock_sha256"] = sha_object(contents)
    filename = out / "STAGE_B_PROTOCOL_LOCK.json"
    if filename.exists():
        existing = json.loads(filename.read_text(encoding="utf-8"))
        if existing != contents:
            raise RuntimeError("Stage B protocol hash/output mismatch. Do not overwrite frozen trial! STOP.")
    else:
        out.mkdir(parents=True, exist_ok=True)
        # Only freeze the protocol after checking all source/data identities.
        filename.write_text(json.dumps(contents, ensure_ascii=False, sort_keys=True, indent=2)+"\n", encoding="utf-8")
    return contents


def safe_import_original(root: Path):
    if str(root / "src") not in sys.path:
        sys.path.insert(0,str(root / "src"))
    if str(root / "experiments") not in sys.path:
        sys.path.insert(0,str(root / "experiments"))
    import dynamic_local_basis_coupling_suite as DYN
    from mode_fri.causal_ssp_modal_tracking import CSMTConfig,CausalSSPModalTracking
    from mode_fri.online_track_reacquire import LocalModeAtomFactory, normalized_projection_scores
    return DYN,CSMTConfig,CausalSSPModalTracking,LocalModeAtomFactory,normalized_projection_scores


def mode_data(t: int, forward: dict) -> tuple[np.ndarray,np.ndarray]:
    n = int(forward["n_modes_per_frame"][t])
    return forward["roots"][t,:n], forward["receiver_mode_shapes"][t,:,:n]


def generated_y(g: str, rep: int, t: int, forward: dict) -> tuple[np.ndarray,dict]:
    roots, b = mode_data(t, forward)
    if np.any(~np.isfinite(b)) or min(np.linalg.norm(b,axis=0)) < 1e-12:
        raise RuntimeError(f"Invalid full KRAKEN mode shapes at {g} frame {t}")
    b = b/np.linalg.norm(b,axis=0,keepdims=True)
    m = len(roots)
    rng_c = np.random.default_rng(stable_seed("NN_FRI_Bv1_C",g,START_SEED+rep,t))
    coef = (rng_c.normal(size=(m,N_SNAPSHOTS)) +1j*rng_c.normal(size=(m,N_SNAPSHOTS)))/math.sqrt(2)
    coef *= (np.arange(1,m+1,dtype=float)**(-MODE_AMPLITUDE_POWER))[:,None]
    clean = b@coef
    rng_n = np.random.default_rng(stable_seed("NN_FRI_Bv1_N",g,START_SEED+rep,t))
    noise = (rng_n.normal(size=clean.shape)+1j*rng_n.normal(size=clean.shape))/math.sqrt(2)
    scale = float(np.linalg.norm(clean)/max(np.linalg.norm(noise),1e-30)/math.sqrt(10**(SNR_DB/10)))
    noise *= scale
    Y = clean +noise
    info={"y_sha256":sha_array(Y),
          "modal_RMS_amplitudes":f"m^-{MODE_AMPLITUDE_POWER}",
          "n_forward_modes":m,
          "achieved_snr_db":10*math.log10(max(np.vdot(clean,clean).real,1e-30)/max(np.vdot(noise,noise).real,1e-30))}
    return Y,info


def step_scores(k: np.ndarray,ssp: np.ndarray,Y: np.ndarray,forward:dict,tools:tuple, *, dz:float)->dict:
    _,_,_,LocalModeAtomFactory,normalized_projection_scores=tools
    factory=LocalModeAtomFactory(frequency_hz=FREQ,ssp=ssp,dz=dz,receiver_depths=forward["receivers"],water_depth=float(forward["H_m"]))
    basis=factory.basis(k)
    score,worst=normalized_projection_scores(Y,basis)
    coef,_,_,_=np.linalg.lstsq(basis,Y,rcond=None)
    resid=Y-basis@coef
    rm=np.linalg.norm(resid,axis=0)/np.maximum(np.linalg.norm(Y,axis=0),1e-30)
    return {"common_mean_squared_resid":float(score),"common_worst_squared_resid":float(worst),
            "common_mean_nonsquared_resid":float(np.mean(rm))}


def row_metrics(k: np.ndarray, root3: np.ndarray, valid_branch: bool)->dict:
    k=np.asarray(k,dtype=float).reshape(-1)
    out={"k1":float(k[0]),"k2":float(k[1]),"k3":float(k[2])}
    if valid_branch:
        err=k-np.real(root3)
        out.update({"model_branch_rmse_rad_m":float(np.sqrt(np.mean(err*err))),
                    "model_branch_mae_rad_m":float(np.mean(np.abs(err))),
                    "model_branch_max_rad_m":float(np.max(np.abs(err)))})
    else:
        out.update({"model_branch_rmse_rad_m":None,"model_branch_mae_rad_m":None,
                    "model_branch_max_rad_m":None})
    return out


def write_csv_atomic(path: Path, rows:list[dict]):
    if not rows: raise ValueError("No rows")
    cols=list(rows[0])
    tmp=path.with_suffix(path.suffix+".partial")
    with tmp.open("w",newline="",encoding="utf-8-sig") as f:
        wr=csv.DictWriter(f,fieldnames=cols)
        wr.writeheader(); wr.writerows(rows)
    os.replace(tmp,path)


def one_case(g:str,rep:int,nframes:int,out:Path,lock:dict,fw:dict,branches:np.ndarray,tools:tuple, *,pilot:bool):
    label=f"{g}_rep{rep:02d}"
    result_csv=out / (f"{label}_PILOT.csv" if pilot else f"{label}.csv")
    audit_json=out / (f"{label}_PILOT.json" if pilot else f"{label}.json")
    if result_csv.exists() and audit_json.exists():
        meta=json.loads(audit_json.read_text(encoding="utf8"))
        if meta.get("lock_sha256")==lock["lock_sha256"] and meta.get("frames")==nframes:
            print(f"reuse checked completed trial: {label}")
            if sha_file(result_csv)!=meta["csv_sha256"]:
                raise RuntimeError("Existing trial CSV tampered; STOP")
            return
        raise RuntimeError("Old trial exists with different protocol/frames; STOP")
    if result_csv.exists() or audit_json.exists():
        raise RuntimeError("Incomplete or unstamped trial; preserve for debugging then remove manually, STOP")
    DYN,Config,Tracker,_,_=tools
    dz=INVERSE_DZ_M[g]; receivers=fw["receivers"].copy();H=float(fw["H_m"])
    zinv=np.arange(round(H/dz)+1,dtype=float)*dz
    if not np.isclose(zinv[-1],H) or np.max(np.abs(receivers/dz-np.round(receivers/dz)))>1e-10:
        raise RuntimeError("Chosen inverse grid does not align exactly with geometry")
    s0=np.interp(zinv,fw["z"],fw["c"][0]).astype(np.float64)
    Y0,info0=generated_y(g,rep,0,fw)
    meta=DYN.DynamicMetadata(K=3,frequency=FREQ,array_geometry=receivers,
                             water_depth=H,SSP=s0)
    t0=time.perf_counter()
    init,_,accepted,coarse,refined=DYN.dynamic_topn_refined(
        Y=Y0, metadata=meta, seed=stable_seed("Bv1_full",g,rep,0),
        cfg=full_params(g),dz=dz)
    init_elapsed=time.perf_counter()-t0
    init=np.sort(np.asarray(init,dtype=float))[::-1]
    if init.shape!=(3,) or not np.all(np.isfinite(init)):
        raise RuntimeError(f"Full initializer failed {label}: {init}")
    trackers={
        "CSMT_dynamic_SSP":(Tracker(Config(**csmt_params(g))),False),
        "CSMT_fixed_SSP":(Tracker(Config(**csmt_params(g))),True),
    }
    if g=="sav15_101":
        trackers["unchanged_local_params_dynamic_SSP_at_101m_only"]=(
            Tracker(Config(**csmt_params(g,unchanged_local=True))),False)
    # Original local config does NOT imply unchanged Full initializer.
    for tr,fixed in trackers.values():
        tr.initialize_from_full_estimate(k_hat=init,SSP_t=s0,Y_t=Y0,
            frequency_hz=FREQ,dz=dz,receiver_depths=receivers,
            water_depth=H,full_runtime_seconds=init_elapsed)
    held=np.asarray(init).copy()
    rows=[]
    for t in range(nframes):
        if t==0:
            Y,gen=Y0,info0
        else:
            Y,gen=generated_y(g,rep,t,fw)
        s_now=np.interp(zinv,fw["z"],fw["c"][t]).astype(float)
        true=fw["roots"][t,:3]
        estimates={
            "hold_k0_fixed_SSP":(held,s0,0.,"HOLD"),
            "hold_k0_dynamic_SSP":(held,s_now,0.,"HOLD"),
        }
        if t==0:
            for name,(trk,fixed) in trackers.items():
                estimates[name]=(trk.state.previous_k.copy(),s0,init_elapsed,"INIT_FULL")
        else:
            for name,(trk,fixed) in trackers.items():
                s=s0 if fixed else s_now
                res=trk.step(Y_t=Y,SSP_t=s,frequency_hz=FREQ,
                             dz=dz,receiver_depths=receivers,water_depth=H)
                estimates[name]=(res.k_hat,s,res.total_runtime_seconds,res.state)
        for name,(estimated,used_ssp,seconds,state) in estimates.items():
            record={"trial":rep,"seed":START_SEED+rep,"geometry":g,"frame":t,
                    "time_min":t*10,"method":name,"state":state,
                    "model_branch_unambiguous":bool(branches[t]),
                    "QC_flagged_temperature":t in (50,95),
                    "n_forward_modes":gen["n_forward_modes"],
                    "y_sha256":gen["y_sha256"],
                    "achieved_snr_db":gen["achieved_snr_db"],
                    "runtime_seconds":float(seconds)}
            record.update(row_metrics(estimated,true,branches[t]))
            record.update(step_scores(estimated,used_ssp,Y,fw,tools,dz=dz))
            rows.append(record)
        if (t+1)%12==0 or t+1==nframes:
            print(f"{label}: {t+1}/{nframes} frames, init runtime={init_elapsed:.2f}s",flush=True)
    # Only completed whole-trial files are committed; never silent partial results.
    write_csv_atomic(result_csv,rows)
    meta={"name":label,"lock_sha256":lock["lock_sha256"],
          "geometry":g,"trial":rep,"frames":nframes,"init_full_runtime_seconds":init_elapsed,
          "init_k":init.tolist(),"init_accepted_refine":bool(accepted),
          "init_coarse_scores":[float(x) for x in coarse],
          "init_refined_scores":[float(x) for x in refined],
          "csv_sha256":sha_file(result_csv),"status":"COMPLETE"}
    tmp=audit_json.with_suffix(".json.partial")
    tmp.write_text(json.dumps(meta,indent=2,sort_keys=True)+"\n",encoding="utf8")
    os.replace(tmp,audit_json)


def summarize(out:Path,p:dict):
    """Only seed-level paired summaries; never treat 144 frames as independent trials."""
    trial_rows=[]
    found=[]
    segments={"full_day":(0,144), "early_48":(0,48),
              "middle_48":(48,96), "late_48":(96,144)}
    for g in GEOM:
        for rep in range(N_REPS):
            f=out / f"{g}_rep{rep:02d}.csv"
            m=out / f"{g}_rep{rep:02d}.json"
            if not f.is_file() or not m.is_file():
                continue
            md=json.loads(m.read_text(encoding="utf8"))
            if sha_file(f)!=md["csv_sha256"]:
                raise RuntimeError(f"Trial checksum fail: {f}")
            if md["frames"]!=144 or md["status"]!="COMPLETE":
                raise RuntimeError(f"Non-final trial in main results: {f}")
            found.append((g,rep))
            d={}
            with f.open("r",encoding="utf-8-sig",newline="") as h:
                for r in csv.DictReader(h):
                    d.setdefault(r["method"],[]).append(r)
            for method,allrows in d.items():
                if len(allrows)!=144 or sorted(int(x["frame"]) for x in allrows)!=list(range(144)):
                    raise RuntimeError(f"Missing method/frame rows in {f}: {method}")
                for seg,(lo,hi) in segments.items():
                    rows=[r for r in allrows if lo<=int(r["frame"])<hi]
                    valid=[r for r in rows if r["model_branch_unambiguous"]=="True" and r["model_branch_rmse_rad_m"]]
                    tracking=[r for r in rows if int(r["frame"])!=0]
                    trial_rows.append({
                        "geometry":g,"seed":START_SEED+rep,"method":method,"segment":seg,
                        "n_frames":len(rows),"n_model_branch_valid":len(valid),
                        "QC_flagged_frames_included":sum(r["QC_flagged_temperature"]=="True" for r in rows),
                        "mean_model_branch_rmse_rad_m":float(np.mean([float(r["model_branch_rmse_rad_m"]) for r in valid])) if valid else float("nan"),
                        "mean_projection_residual_nonsquared":float(np.mean([float(r["common_mean_nonsquared_resid"]) for r in rows])),
                        "mean_projection_residual_squared":float(np.mean([float(r["common_mean_squared_resid"]) for r in rows])),
                        "mean_runtime_seconds_excluding_initial_Full":float(np.mean([float(r["runtime_seconds"]) for r in tracking])) if tracking else float("nan")})
    if trial_rows:
        write_csv_atomic(out/"COMPLETED_SEED_LEVEL_SEGMENT_SUMMARIES.csv",trial_rows)
    complete={g:sum(x[0]==g for x in found) for g in GEOM}
    print("COMPLETE trials by geometry: "+str(complete))
    if len(found)<N_REPS*len(GEOM):
        print("INCOMPLETE: do not report final 20-seed paired statistics")
        return
    agg={}
    lookup={}
    for r in trial_rows:
        agg.setdefault((r["geometry"],r["method"],r["segment"]),[]).append(r)
        lookup[(r["geometry"],r["segment"],r["seed"],r["method"])]=r
    final=[]
    for (g,method,seg),ar in sorted(agg.items()):
        if len(ar)!=N_REPS:
            raise RuntimeError(f"Expected 20 seed-level rows for {g},{method},{seg}; got {len(ar)}")
        final.append({"geometry":g,"method":method,"segment":seg,
            "n_independent_seeds":len(ar),
            "mean_seed_level_model_branch_rmse":float(np.mean([x["mean_model_branch_rmse_rad_m"] for x in ar])),
            "mean_seed_level_projection_residual":float(np.mean([x["mean_projection_residual_nonsquared"] for x in ar])),
            "mean_seed_level_tracking_runtime_s":float(np.mean([x["mean_runtime_seconds_excluding_initial_Full"] for x in ar]))})
    write_csv_atomic(out/"STAGE_B_COMPLETE_20SEED_DESCRIPTIVE.csv",final)
    pairs=[("dynamic_minus_fixed","CSMT_dynamic_SSP","CSMT_fixed_SSP"),
           ("dynamic_minus_held_dynamic","CSMT_dynamic_SSP","hold_k0_dynamic_SSP"),
           ("fixed_minus_held_fixed","CSMT_fixed_SSP","hold_k0_fixed_SSP"),
           ("unchanged_local_minus_transferred_dynamic_at_101m",
            "unchanged_local_params_dynamic_SSP_at_101m_only","CSMT_dynamic_SSP")]
    stats=[]
    for g in GEOM:
        for seg in segments:
            for label,method_a,method_b in pairs:
                if label.startswith("unchanged_local") and g!="sav15_101":
                    continue
                for metric in ("mean_model_branch_rmse_rad_m","mean_projection_residual_nonsquared"):
                    diffs=np.asarray([
                        lookup[(g,seg,START_SEED+rep,method_a)][metric]-
                        lookup[(g,seg,START_SEED+rep,method_b)][metric]
                        for rep in range(N_REPS)],dtype=float)
                    if not np.all(np.isfinite(diffs)):
                        raise RuntimeError(f"Nonfinite seed-level paired difference for {g},{seg},{label},{metric}")
                    rng=np.random.default_rng(stable_seed(
                        "SAVEX15_Bv1_20260930_paired_bootstrap",g,seg,label,metric))
                    idx=rng.integers(0,N_REPS,size=(10000,N_REPS))
                    boot=np.mean(diffs[idx],axis=1)
                    stats.append({"geometry":g,"segment":seg,"contrast_A_minus_B":label,
                        "metric":metric,"n_paired_seeds":N_REPS,"mean_paired_difference":float(np.mean(diffs)),
                        "bootstrap_95_CI_low":float(np.quantile(boot,0.025)),
                        "bootstrap_95_CI_high":float(np.quantile(boot,0.975)),
                        "bootstrap_unit":"independent coefficient/noise seed, fixed observed 1-day environment"})
    write_csv_atomic(out/"STAGE_B_COMPLETE_PAIRED_SEED_BOOTSTRAP.csv",stats)
    print("Both geometries, 20 paired seeds each complete. Descriptive and bootstrap tables saved.")


def export_min(out:Path):
    """Only completed 20-seed results; Stage A independent archive remains separate."""
    run = out/"RUN_MAIN"
    expected=[]
    for g in GEOM:
        for rep in range(N_REPS):
            stem=f"{g}_rep{rep:02d}"
            a,b=run/f"{stem}.csv",run/f"{stem}.json"
            if not a.is_file() or not b.is_file():
                raise RuntimeError(f"40-trial export requires all runs. Missing: {a} or {b}")
            d=json.loads(b.read_text(encoding="utf8"))
            if d["status"]!="COMPLETE" or d["frames"]!=144 or sha_file(a)!=d["csv_sha256"]:
                raise RuntimeError(f"Unvalidated or incomplete trial: {stem}")
            expected.extend([a,b])
    for n in ("COMPLETED_SEED_LEVEL_SEGMENT_SUMMARIES.csv",
              "STAGE_B_COMPLETE_20SEED_DESCRIPTIVE.csv",
              "STAGE_B_COMPLETE_PAIRED_SEED_BOOTSTRAP.csv"):
        f=run/n
        if not f.is_file():
            raise RuntimeError(f"Run --summarize after all 40 trials before export: {f}")
        expected.append(f)
    expected.append(out/"STAGE_B_PROTOCOL_LOCK.json")
    expected.append(Path(__file__).resolve())
    zpath=out/"NN_FRI_SAVEX15_StageB_RESULT_EXPORT.zip"
    if zpath.exists():
        raise FileExistsError(f"Export already exists, never overwrite: {zpath}")
    manifest={}
    with zipfile.ZipFile(zpath,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for f in expected:
            if f==Path(__file__).resolve():
                arc="SOURCE_SCRIPT/"+f.name
            else:
                arc=f.relative_to(out).as_posix()
            z.write(f,arcname=arc)
            manifest[arc]=sha_file(f)
        z.writestr("STAGE_B_EXPORT_SHA256.json",json.dumps(manifest,indent=2,sort_keys=True))
        z.writestr("BUNDLE_LIMITATIONS.txt",
            "20 paired synthetic-acoustic noise/coeff seeds on ONE real observed-temperature day.\n"
            "Not 20 independent ocean environments or SAVEX real acoustic measurements.\n"
            "KRAKEN seabed assumed; mode coefficients idealized; data contain QC-flagged frames 50/95.\n"
            "Full NNFR only at first frame; no every-frame Full benchmark.\n")
    print("READY FOR AUDIT:",zpath,"bytes=",zpath.stat().st_size)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project-root",type=Path,required=True,help="original frozen NN_FRI_Mode_Extension checkout")
    ap.add_argument("--stagea-root",type=Path,required=True,help="local Stage A outputs/savex15_stageA_physics_repaired_v1")
    ap.add_argument("--out",type=Path,default=None)
    group=ap.add_mutually_exclusive_group(required=True)
    for a in ("preflight","pilot","run","summarize","export_min"):
        group.add_argument("--"+a,action="store_true")
    ap.add_argument("--pilot-geometry",choices=list(GEOM),default="sav15_101")
    ap.add_argument("--trial-count",type=int,default=N_REPS,help="run 1..20 predeclared seeds, same frozen protocol")
    args=ap.parse_args()
    root=args.project_root.resolve();stagea=args.stagea_root.resolve()
    out=(args.out.resolve() if args.out is not None else (stagea.parent/"savex15_stageB_v1").resolve())
    source_hashes=verify_sources(root)
    data_lock,forward=load_stagea(stagea)
    p=protocol()
    branch_data={g:check_branches(f,g) for g,f in forward.items()}
    lock=verify_lock(out,p,data_lock,forward)
    print("FROZEN CHECK: PASS; Stage A unchanged; source hashes and Stage B protocol locked")
    print("Geometry-only transfer ratios:", {g:round(homogeneous_scale(GEOM[g]),6) for g in GEOM})
    print("Full coarse minseps:",{g:full_params(g)["grid_min_separation_rad_m"] for g in GEOM})
    print("Tracker minseps:",{g:csmt_params(g)["minimum_mode_separation_rad_m"] for g in GEOM})
    print("Model branch preflight:",[info for mask,info in branch_data.values()])
    if args.preflight:
        return
    if args.summarize:
        summarize(out/"RUN_MAIN",p);return
    if args.export_min:
        export_min(out);return
    tools=safe_import_original(root)
    if args.pilot:
        path=out/"PILOT_NOT_FOR_FINAL";path.mkdir(parents=True,exist_ok=True)
        g=args.pilot_geometry
        one_case(g,0,3,path,lock,forward[g],branch_data[g][0],tools,pilot=True)
        print("PILOT finished. Inspect outputs before attempting 20-seed full runs.")
        return
    if not 1<=args.trial_count<=N_REPS:raise ValueError("trial count must be 1..20")
    path=out/"RUN_MAIN";path.mkdir(parents=True,exist_ok=True)
    for g in GEOM:
        for rep in range(args.trial_count):
            one_case(g,rep,144,path,lock,forward[g],branch_data[g][0],tools,pilot=False)
    summarize(path,p)


if __name__=="__main__":
    main()
