#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NN_FRI SAVEX15, Phase A: auditable *environment-only* repair.

DOES NOT run CSMT/NNFR and DOES NOT re-use old SAVEX numerical results.
The fixed procedure and assumptions are in EXPERIMENT_LOCK.json, generated during prepare.

Inputs: previously QC-ed real profile T/SP/metadata, 71098 thermistor temperatures.
Outputs: 144 timestamps x 2 synthetic geometry ENV cases and auditable TEOS-10 SSPs;
         optional KRAKEN execution and archived .MODs.
All prior results remain untouched: no overwrite of previously prepared changed .ENV.

Require: numpy, pandas, gsw, installed KRAKEN for --run-kraken.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

DAY = dt.datetime(2015, 5, 20, tzinfo=dt.timezone.utc)
END = DAY + dt.timedelta(days=1)
FREQ_HZ = 300.0
INTERVAL_SECONDS = 600
BOTTOM = {"description": "ASSUMED, not a site-measured geoacoustic model",
          "cp_m_s": 1650.0, "rho_g_cm3": 1.9, "atten_db_per_wavelength": 0.5}
CASES = {"sav15_101": {"H": 101.0, "dz": 0.25, "receiver_m": np.linspace(30, 90, 21)},
         "design_40": {"H": 40.0, "dz": 0.05, "receiver_m": np.arange(1, 41, 1.0)}}
# Set BEFORE running results: select the most recent, nearby, deep (>95m) pre-day CTD.
BASELINE = {"pre_event_cutoff_utc": DAY.isoformat(), "profile_source": "CTD",
            "min_max_depth_m": 95.0, "max_distance_mooring_km": 5.0,
            "selection": "most recent qualifying CTD, tie-break by distance then profile_id",
            "missing_bottom_T_SP": "constant extension from deepest CTD measurement",
            "mooring_temperature": "use only each timestamp; interpolate vertically between valid same-time sensors",
            "duplicate_sensor_depth": "mean available same-timestamp temperature at repeated depth",
            "salinity": "static in-situ practical salinity from selected pre-day CTD",
            "thermodynamic_definition": "GSW SA_from_SP -> CT_from_t -> sound_speed",
            "note": "time-causal wrt salinity baseline only; synthetic forward from real observed thermistor temperatures"}


def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''): h.update(b)
    return h.hexdigest()


def dump_json(p: Path, obj: dict):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(obj,indent=2,ensure_ascii=False,sort_keys=True,default=str)+'\n',encoding='utf-8')


def immutable_text(p: Path, value: str):
    if p.is_file():
        if p.read_text(encoding='utf-8') != value:
            raise RuntimeError(f'Output conflict; NEVER overwrite changed file. Use a fresh --out: {p}')
    else:
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(value,encoding='utf-8',newline='\n')


def find_inputs(project:Path):
    qc=project/'outputs'/'savex15_qc_v1'
    f={"profile_summary":qc/'profiles_summary.csv',
       "profile_levels":qc/'profiles_levels.csv',
       "mooring":qc/'mooring_71098.npz'}
    missing=[str(v) for v in f.values() if not v.is_file()]
    if missing: raise FileNotFoundError('Missing QC inputs (do not regenerate old results):\n'+'\n'.join(missing))
    provenance=project/'INPUT_PROVENANCE.json'
    if provenance.is_file():
        dat=json.loads(provenance.read_text(encoding='utf-8'))
        for key,path in f.items():
            if sha(path)!=dat['sha256'][key]:
                raise RuntimeError(f'Input provenance SHA256 mismatch for {key}; STOP: {path}')
    return f


def select_pre_event_ctd(summary: pd.DataFrame,lat:float,lon:float):
    x=summary.copy()
    x['when']=pd.to_datetime(x['time_utc'],utc=True,errors='coerce')
    x['distance_km']=( (111.2*(x['lat']-lat))**2 +
                       (111.2*np.cos(np.deg2rad(lat))*(x['lon']-lon))**2 )**.5
    qualifies=((x['source']=='CTD')&(x['qc']=='PASS')&(x['when']<pd.Timestamp(DAY))&
      (x['p_max']>=BASELINE['min_max_depth_m'])&
      (x['distance_km']<=BASELINE['max_distance_mooring_km']))
    eligible=x.loc[qualifies].sort_values(['when','distance_km','profile_id'],ascending=[False,True,True])
    if eligible.empty: raise RuntimeError('No eligible pre-event local deep CTD; STOP rather than use future data.')
    return eligible.iloc[0],eligible


def validate_chain(z:np.ndarray,temp:np.ndarray,flags:np.ndarray,t:np.ndarray):
    if z.shape!=(30,) or temp.shape!=(len(t),30) or flags.shape!=temp.shape:
        raise RuntimeError(f'Unexpected mooring dimensions: depths{z.shape}, T{temp.shape}, QC{flags.shape}')
    if not np.all(np.diff(t)>0): raise RuntimeError('Nonmonotonic mooring time; STOP')
    inds=np.where((t>=DAY.timestamp())&(t<END.timestamp()))[0]
    if len(inds)!=2880: raise RuntimeError(f'Expected all 2880 day records, got {len(inds)}; STOP')
    # This is an expressly PREDECLARED 00:00 - 23:50 UTC every 10 minutes.
    picks=inds[::INTERVAL_SECONDS//30]
    if len(picks)!=144: raise RuntimeError(f'Expected 144 decimated records, got {len(picks)}')
    target=DAY.timestamp()+np.arange(144)*INTERVAL_SECONDS
    if np.max(np.abs(t[picks]-target))>1.0:
        raise RuntimeError('Chain timestamps not aligned to UTC day every 10 min (1 s tolerance)')
    return picks


def load_state(project:Path):
    paths=find_inputs(project)
    d=np.load(paths['mooring'],allow_pickle=False)
    mt=np.asarray(d['time_unix_seconds'],float)
    mz=np.asarray(d['depth'],float)
    mT=np.asarray(d['temperature'],float)
    mflags=np.asarray(d['flags'])
    lat,lon=float(d['lat']),float(d['lon'])
    if not (31<=lat<=34 and 125<=lon<=127.5): raise RuntimeError('Unusual mooring location')
    picks=validate_chain(mz,mT,mflags,mt)
    summ=pd.read_csv(paths['profile_summary'])
    choice,elig=select_pre_event_ctd(summ,lat,lon)
    lev=pd.read_csv(paths['profile_levels'],usecols=['profile_id','depth_m','temperature_c','salinity_psu','level_flags'])
    base=lev.loc[(lev.profile_id==choice.profile_id)&(lev.level_flags==0),
        ['depth_m','temperature_c','salinity_psu']].sort_values('depth_m')
    base=base.groupby('depth_m',as_index=False).median(numeric_only=True)
    zz=np.asarray(base.depth_m,float)
    TT=np.asarray(base.temperature_c,float)
    SP=np.asarray(base.salinity_psu,float)
    if len(base)<60 or zz.min()>3 or zz.max()<95 or np.any(np.diff(zz)<=0):
        raise RuntimeError('Selected CTD has insufficient clean full-depth measurement: STOP')
    if not (30<np.nanmedian(SP)<37 and 0<np.nanmedian(TT)<30):
        raise RuntimeError('Unexpected CTD SP/T values')
    return paths,dict(lat=lat,lon=lon,time=mt,z_chain=mz,T_chain=mT,
        flags=mflags,picks=picks,ctd=choice,eligible=elig,z_ctd=zz,T_ctd=TT,SP_ctd=SP)


def construct_one(st:dict,z:np.ndarray,i:int):
    # Interpolate fixed pre-event CTD only; constant top/bottom extrapolation is explicit.
    T=np.interp(z,st['z_ctd'],st['T_ctd']); SP=np.interp(z,st['z_ctd'],st['SP_ctd'])
    usable=(st['flags'][i]==0)&np.isfinite(st['T_chain'][i])&np.isfinite(st['z_chain'])
    if int(usable.sum())<20: raise RuntimeError(f'Insufficient observed mooring temperatures at timestep {i}')
    moor=pd.DataFrame({'z':st['z_chain'][usable], 'T':st['T_chain'][i,usable]})
    # 22m has TWO separate thermistors; combine available same-time measurements.
    moor=moor.groupby('z',as_index=False,sort=True).mean(numeric_only=True)
    mz,mt=moor['z'].to_numpy(float),moor['T'].to_numpy(float)
    if len(mz)<18 or (mz[-1]-mz[0])<65: raise RuntimeError('Insufficient mooring depth span')
    observed_band=(z>=mz[0])&(z<=mz[-1]);T[observed_band]=np.interp(z[observed_band],mz,mt)
    return T,SP,{"n_valid_physical_thermistors":int(usable.sum()),
                 "n_distinct_valid_depths":int(len(mz)),
                 "observed_min_depth_m":float(mz[0]),"observed_max_depth_m":float(mz[-1]),
                 "n_flagged_sensors":int(30-usable.sum())}


def teos(z:np.ndarray,T:np.ndarray,SP:np.ndarray,lat:float,lon:float):
    try: import gsw
    except ImportError as e: raise RuntimeError('GSW is REQUIRED for repaired physical SSP: pip install gsw') from e
    # T is observed *in-situ* thermometer temperature; SP is practical salinity, NOT SA.
    p=gsw.p_from_z(-z,lat)
    SA=gsw.SA_from_SP(SP,p,np.full_like(z,lon),np.full_like(z,lat))
    CT=gsw.CT_from_t(SA,T,p)
    c=np.asarray(gsw.sound_speed(SA,CT,p),dtype=float)
    if not np.all(np.isfinite(c)) or np.any((c<1430)|(c>1570)):
        raise RuntimeError('Physically anomalous TEOS sound-speed column; STOP')
    return c


def env_body(name:str,cfg:dict,z:np.ndarray,c:np.ndarray):
    # Same declared hypothetical sandy halfspace as historical Plan-A: no tuned bottom.
    H=float(cfg['H']); r=np.asarray(cfg['receiver_m'],float)
    out=[f"'SAVEX15_REPAIR_{name}'",f'{FREQ_HZ:.4f}', '1',"'CVW'",f'2000 0.0 {H:.4f}']
    out += [f'{zi:.4f} {ci:.6f} 0.0 1.000 0.0 0.0 /' for zi,ci in zip(z,c)]
    out += ["'A' 0.0",f"{H:.4f} {BOTTOM['cp_m_s']:.4f} 0.0 {BOTTOM['rho_g_cm3']:.3f} {BOTTOM['atten_db_per_wavelength']:.3f} 0.0 /",
            f"100.0 {BOTTOM['cp_m_s']*1.03:.4f}", '10.0', '1', '20.0 /',str(len(r)),
            ' '.join(f'{v:.4f}' for v in r)+' /']
    return '\n'.join(out)+'\n'


def prepare(project:Path,out:Path):
    paths,st=load_state(project)
    try: import gsw
    except ImportError as e: raise RuntimeError('Please install gsw in your project .venv: pip install gsw') from e
    manifest=out/'EXPERIMENT_LOCK.json'
    ctd=st['ctd']
    uniq,counts=np.unique(st['z_chain'],return_counts=True)
    duplicates=uniq[counts>1].tolist()
    lock={"version":"SAVEX15_STAGE_A_PHYSICS_v1","status":"ENV_READY_ONLY",
        "data_classification":"REAL observed T/SP forcing + independent synthetic acoustic forward; NOT field acoustic validation",
        "design_date_utc":DAY.isoformat(),"n_time_steps":144,"sample_minutes":10,"frequency_hz":FREQ_HZ,
        "code_sha256":sha(Path(__file__).resolve()),"gsw_version":getattr(gsw,'__version__','unknown'),
        "input_sha256":{k:sha(v) for k,v in paths.items()},
        "input_paths":{k:str(v.resolve()) for k,v in paths.items()},
        "baseline_rule":BASELINE,"bottom_model":BOTTOM,
        "mooring":{"lat":st['lat'],"lon":st['lon'],"depth_sensor_count":len(st['z_chain']),
                   "duplicate_depths_m":duplicates},
        "pre_event_ctd":{"id":str(ctd.profile_id),"time_utc":str(ctd.time_utc),
           "lat":float(ctd.lat),"lon":float(ctd.lon),"distance_km":float(ctd.distance_km),
           "n_eligible":len(st['eligible']),"measured_max_depth_m":float(st['z_ctd'].max()),
           "metadata_note":"CHOSEN BEFORE reading KRAKEN outputs; not selected for performance"},
        "cases":{k:{"H_m":v['H'],"grid_dz_m":v['dz'],"receiver_depths_m":v['receiver_m'].tolist(),
                     "geometry_type":"counterfactual design point" if k=='design_40' else 'SAVEX-geometry-inspired'} for k,v in CASES.items()},
        "synthesis_boundary":"SAVEX chain 2-80 m supplies time variation; static pre-day CTD fills unmeasured water",
        "limitations":["SP is temporally fixed from CTD; dynamic only at chain's measured temperature depths",
                        "Selected CTD reflects pre-day water, not exact every-frame ambient salinity",
                        "Assumed seabed is not field-validated", "SAVEX acoustic recordings are NOT used"]}
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf8'))
        # If a source changed, do NOT use existing output folder; source identity matters.
        for key in ['version','code_sha256','gsw_version','input_sha256','baseline_rule','bottom_model','pre_event_ctd','cases']:
            if old.get(key)!=lock[key]: raise RuntimeError(f'Prior EXPERIMENT_LOCK conflict on {key}; use fresh --out')
    else:
        dump_json(manifest,lock)
    meta=[]
    for tag,cfg in CASES.items():
        z=np.linspace(0,float(cfg['H']),int(round(cfg['H']/cfg['dz']))+1)
        C=np.empty((144,len(z)),dtype=float)
        for j,i in enumerate(st['picks']):
            T,SP,qc=construct_one(st,z,int(i)); C[j]=teos(z,T,SP,st['lat'],st['lon'])
            case=f'{tag}_{j:03d}'
            fp=out/'cases'/tag/case/(case+'.env')
            immutable_text(fp,env_body(case,cfg,z,C[j]))
            meta.append({"case":case,"tag":tag,"frame":j,"time_utc":dt.datetime.fromtimestamp(st['time'][i],dt.timezone.utc).isoformat(),
                         "env_relpath":fp.relative_to(out).as_posix(),"env_sha256":sha(fp),
                         "c_min_m_s":float(C[j].min()),"c_max_m_s":float(C[j].max()), **qc})
        dest=out/f'SSP_INPUT_{tag}.npz'
        if dest.exists():
            with np.load(dest,allow_pickle=False) as prev:
                if not(np.array_equal(prev['c'],C) and np.array_equal(prev['z'],z)):
                    raise RuntimeError(f'Existing SSP input conflicts: {dest}; use fresh --out')
        else:
            np.savez_compressed(dest, z=z,c=C,t_unix=st['time'][st['picks']],
                receiver_depths_m=cfg['receiver_m'],frequency_hz=FREQ_HZ,H_m=cfg['H'],
                pre_event_ctd_id=str(ctd.profile_id))
        lock[f'ssp_npz_sha256_{tag}']=sha(dest)
    meta_path=out/'CASES_PREDECLARED.csv'
    csv_text=pd.DataFrame(meta).to_csv(index=False)
    immutable_text(meta_path,csv_text)
    lock['case_manifest_sha256']=sha(meta_path)
    # Distinguish immutable fixed fields from recorded, derived hashes.
    if manifest.exists():
        old=json.loads(manifest.read_text(encoding='utf8'))
        for k in ('case_manifest_sha256',*[f'ssp_npz_sha256_{x}' for x in CASES]):
            if k in old and old[k]!=lock[k]: raise RuntimeError(f'Existing manifest derived value inconsistent: {k}')
    dump_json(manifest,lock)
    print(f'PREPARED 288 environment cases; eligible CTDs={len(st["eligible"])}; chosen {ctd.profile_id} ({ctd.time_utc})')
    print(f'Output: {out}\nNext: --run-kraken --kraken-exe ... (after checking lock and one representative .env)')


def read_mod(path:Path,receiver_depths:np.ndarray,frequency_hz=FREQ_HZ,max_phase_c=1850.0):
    """Fail closed for the *previously audited* little-endian KRAKEN direct-access .MOD layout.

    If this build of KRAKEN uses a different binary layout, collect one MOD/PRT pair
    for parser audit; NEVER silently parse unsupported layouts.
    """
    b=path.read_bytes()
    if len(b)<4:raise RuntimeError(f'Empty MOD {path}')
    words=int(np.frombuffer(b[:4],dtype='<i4')[0]); L=words*4
    if not(10<=words<=10000) or len(b)%L!=0:raise RuntimeError(f'Unsupported .MOD record size {path}')
    nrec=len(b)//L
    rec=lambda j:b[(j-1)*L:j*L]
    nmode=int(np.frombuffer(rec(6)[:4],dtype='<i4')[0])
    if not(3<=nmode<=100) or nrec!=nmode+8:raise RuntimeError(f'Unsupported .MOD record layout nmode={nmode} records={nrec}')
    # KRAKEN sorts source & receiver depths. Offset of source may vary by output build.
    zz=np.frombuffer(rec(5),dtype='<f4').astype(float)
    used=[]
    for z in receiver_depths:
        matches=np.flatnonzero(np.isclose(zz,z,atol=0.004,rtol=0))
        if len(matches)!=1:raise RuntimeError(f'Unexpected or duplicate saved receiver depth {z}, matches={len(matches)}')
        used.append(int(matches[0]))
    nmstored=max(used)+1
    if nmstored*8>L:raise RuntimeError('Too many saved depths for MOD record length')
    kk=np.frombuffer(rec(8+nmode),dtype='<c8',count=nmode).astype(np.complex128)
    shapes=np.column_stack([np.frombuffer(rec(8+j),dtype='<c8',count=nmstored).astype(np.complex128)
                            for j in range(nmode)])
    kk,shapes=kk[np.argsort(kk.real)[::-1]], shapes[:,np.argsort(kk.real)[::-1]]
    vr=shapes[used,:]
    cp=2*np.pi*frequency_hz/kk.real
    if np.any(~np.isfinite(kk)) or np.any(cp<1350) or np.any(cp>max_phase_c) or np.any(np.diff(kk.real)>=0):
        raise RuntimeError('Abnormal modal root spectrum; STOP and inspect MOD/PRT')
    if np.any(~np.isfinite(vr)) or np.any(np.linalg.norm(vr,axis=0)<1e-15):
        raise RuntimeError('Invalid zero/nonfinite receiver mode shape')
    return kk,vr,{"record_length_bytes":L,"n_records":nrec,"n_modes":nmode}


def exe_env(exe:Path):
    env=os.environ.copy()
    if os.name=='nt':
        p=Path(r'D:\msys64\mingw64\bin')
        if p.is_dir(): env['PATH']=str(p)+os.pathsep+env.get('PATH','')
    if not exe.is_file(): raise FileNotFoundError(f'KRAKEN executable not found: {exe}')
    return env


def run_kraken(out:Path,exe:Path,limit:int|None=None):
    lockfile=out/'EXPERIMENT_LOCK.json'
    if not lockfile.exists(): raise RuntimeError('First --prepare-only to lock physical inputs')
    lock=json.loads(lockfile.read_text(encoding='utf-8'))
    # Code version is a provenance check but source may be copied/renamed; lock to content SHA.
    if lock['code_sha256']!=sha(Path(__file__).resolve()):raise RuntimeError('Code changed after --prepare-only; use original file or prepare anew')
    for k,expected in lock['input_sha256'].items():
        # Do not permit changing source/input while running.
        src=Path(lock['input_paths'][k]) if lock.get('input_paths') else None
        if not src or not src.is_file() or sha(src)!=expected:raise RuntimeError(f'QC source missing/changed: {k}')
    register=pd.read_csv(out/'CASES_PREDECLARED.csv')
    if sha(out/'CASES_PREDECLARED.csv')!=lock['case_manifest_sha256']:
        raise RuntimeError('Case register has changed, STOP')
    rows=register.to_dict('records')
    if limit is not None: rows=rows[:limit]
    env=exe_env(exe)
    for idx,row in enumerate(rows,1):
        base=out/row['env_relpath']; parent=base.parent; name=row['case']
        mod=parent/(name+'.mod'); prt=parent/(name+'.prt');stamp=parent/'KRAKEN_RUN.json'
        if not base.is_file() or sha(base)!=row['env_sha256']:raise RuntimeError(f'Changed env: {base}')
        if mod.exists() and stamp.exists():
            previous=json.loads(stamp.read_text(encoding='utf-8'))
            if previous['env_sha256']!=sha(base) or previous['mod_sha256']!=sha(mod):
                raise RuntimeError(f'Inconsistent existing result for {name}')
            # Always parse existing modes too, so broken archived result is not hidden.
            read_mod(mod,CASES[row['tag']]['receiver_m'])
            print(f'[{idx}/{len(rows)}] Reuse checked: {name}',flush=True)
            continue
        if mod.exists() or prt.exists():raise RuntimeError(f'Unstamped old result {name}; STOP')
        print(f'[{idx}/{len(rows)}] KRAKEN {name}',flush=True)
        t0=time.perf_counter()
        p=subprocess.run([str(exe),name],cwd=parent,env=env,capture_output=True,text=True,
                         errors='replace',timeout=600)
        (parent/'solver_stdout.txt').write_text(p.stdout or '',encoding='utf8')
        (parent/'solver_stderr.txt').write_text(p.stderr or '',encoding='utf8')
        if p.returncode or not mod.is_file() or not prt.is_file():
            raise RuntimeError(f'KRAKEN failed case {name}, exit={p.returncode}; upload .env/.prt/stdout/stderr for debugging')
        pt=prt.read_text(encoding='utf-8',errors='replace')
        if re.search(r'FATAL\s+ERROR|ERROR\s+STOP',pt,re.I):raise RuntimeError(f'PRT fatal message {name}')
        roots,shapes,meta=read_mod(mod,CASES[row['tag']]['receiver_m'])
        dump_json(stamp,{"name":name,"env_sha256":sha(base),"mod_sha256":sha(mod),
          "prt_sha256":sha(prt),"kraken_exe_sha256":sha(exe),"elapsed_seconds":time.perf_counter()-t0,
          "n_modes":len(roots),"mod_layout":meta})
    # End-to-end aggregation only when EVERY predeclared case exists.
    done=sum(int((out/Path(x['env_relpath']).parent/'KRAKEN_RUN.json').is_file()) for x in register.to_dict('records'))
    print(f'Solver stamps: {done}/288')
    if done==len(register): assemble(out)


def assemble(out:Path):
    lock=json.loads((out/'EXPERIMENT_LOCK.json').read_text(encoding='utf8'))
    manifest=pd.read_csv(out/'CASES_PREDECLARED.csv')
    if len(manifest)!=288:raise RuntimeError('Expected 288 cases')
    for tag,cfg in CASES.items():
        rows=manifest[manifest.tag==tag].sort_values('frame')
        if list(rows.frame)!=list(range(144)):raise RuntimeError(f'Missing or unordered cases for {tag}')
        kk=[]; sr=[]; meta=[]
        for row in rows.to_dict('records'):
            env=out/row['env_relpath']; d=env.parent; name=row['case'];
            stamp=json.loads((d/'KRAKEN_RUN.json').read_text(encoding='utf-8'))
            mod=d/(name+'.mod')
            if sha(env)!=stamp['env_sha256'] or sha(mod)!=stamp['mod_sha256']:
                raise RuntimeError(f'Tampered KRAKEN result: {name}')
            k,shape,md=read_mod(mod,cfg['receiver_m'])
            kk.append(k);sr.append(shape);meta.append(md)
        counts=np.asarray([len(v) for v in kk],dtype=np.int32)
        if np.min(counts)<3:raise RuntimeError(f'Fewer than three model roots in some frames: {tag}; STOP')
        max_modes=int(np.max(counts))
        # Preserve all roots/mode shapes; a mode crossing the phase-speed cutoff
        # is scientific information, NOT a reason for outcome-dependent trimming.
        roots=np.full((144,max_modes),np.nan+1j*np.nan,dtype=np.complex128)
        modes=np.zeros((144,len(cfg['receiver_m']),max_modes),dtype=np.complex128)
        for j,(kj,phi) in enumerate(zip(kk,sr)):
            roots[j,:len(kj)]=kj
            modes[j,:,:phi.shape[1]]=phi
        dest=out/f'KRAKEN_INDEPENDENT_FORWARD_{tag}.npz'
        with np.load(out/f'SSP_INPUT_{tag}.npz',allow_pickle=False) as inputs:
            data={k:inputs[k] for k in ['c','z','t_unix']}
        if dest.exists():
            with np.load(dest,allow_pickle=False) as prev:
                if not(np.array_equal(prev['roots'],roots,equal_nan=True) and
                       np.array_equal(prev['receiver_mode_shapes'],modes) and
                       np.array_equal(prev['n_modes_per_frame'],counts)):
                    raise RuntimeError(f'Previously aggregated result changed: {dest}')
        else:
            np.savez_compressed(dest,roots=roots,receiver_mode_shapes=modes,
                n_modes_per_frame=counts,
                c=data['c'],z=data['z'],t_unix=data['t_unix'],
                receivers=cfg['receiver_m'],freq_hz=FREQ_HZ,H_m=cfg['H'])
        print(f'{tag}: 144 frames; modeled roots/frame {counts.min()}..{counts.max()};'
              ' independent KRAKEN shapes saved. Branch continuity NOT YET proven.',flush=True)

    print('COMPLETE: Stage-A environment and solver outputs. Do not report original Plan-A RMSE as a repaired result.')


def inspect(project:Path):
    _,st=load_state(project)
    print('QC input inspection (NO TEOS or KRAKEN calculation):')
    print(f"  mooring ({st['lat']:.5f}, {st['lon']:.5f}); samples={len(st['time'])}; day picks={len(st['picks'])}")
    print(f"  eligible pre-day CTDs = {len(st['eligible'])}")
    q=st['ctd'];print(f'  deterministic selected CTD: {q.profile_id}  {q.time_utc} distance={q.distance_km:.3f}km')
    print(f"  clean CTD levels={len(st['z_ctd'])}, zmax={st['z_ctd'].max():.2f}m")
    rows=[]
    for j,i in enumerate(st['picks']):
        _,_,r=construct_one(st,np.arange(0.,101.01,1.),int(i))
        rows.append(r)
    print(f"  144 picks: valid thermistor range {min(x['n_valid_physical_thermistors'] for x in rows)}-"+
          f"{max(x['n_valid_physical_thermistors'] for x in rows)} / 30; QC-flagged steps " +
          str(sum(x['n_flagged_sensors']>0 for x in rows)))
    print('  IMPORTANT: no revised TEOS calculation has been performed in this mode.')



def export_min(out:Path):
    """Export traceable all-case ENV+MOD+stamps and archived forward arrays; omit bulky .PRT except spot examples."""
    lockfile=out/'EXPERIMENT_LOCK.json'
    if not lockfile.exists():raise RuntimeError('Missing EXPERIMENT_LOCK.json')
    register=pd.read_csv(out/'CASES_PREDECLARED.csv')
    if len(register)!=288:raise RuntimeError('Expected 288 cases for final audit package')
    required=[lockfile,out/'CASES_PREDECLARED.csv',Path(__file__).resolve()]
    for tag in CASES:
        required.extend([out/f'SSP_INPUT_{tag}.npz',out/f'KRAKEN_INDEPENDENT_FORWARD_{tag}.npz'])
    for row in register.to_dict('records'):
        env=out/row['env_relpath'];d=env.parent;name=row['case']
        required.extend([env,d/(name+'.mod'),d/'KRAKEN_RUN.json'])
        if row['frame'] in (0,143): required.append(d/(name+'.prt'))
    for x in required:
        if not x.is_file():raise FileNotFoundError(f'Missing final export requirement: {x}')
    zip_path=out/'NN_FRI_SAVEX15_StageA_AUDIT_EXPORT.zip'
    if zip_path.exists():raise FileExistsError(f'Audit export already exists; leave old copy intact: {zip_path}')
    with zipfile.ZipFile(zip_path,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
        for p in required:
            arc='SOURCE_SCRIPT/'+p.name if p==Path(__file__).resolve() else p.relative_to(out).as_posix()
            z.write(p,arcname=arc)
        z.writestr('BUNDLE_README.txt',
            'STAGE A only: correct thermodynamic input and environment KRAKEN reconstruction.\n'
            'NO new CSMT performance conclusions. SAVEX observations are T/SP, not acoustic.\n'
            'SHA256s in EXPERIMENT_LOCK and per-case KRAKEN_RUN.json.\n')
    print(f'AUDIT EXPORT: {zip_path}  ({zip_path.stat().st_size/1e6:.2f}MB)')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project-root',type=Path,default=Path(__file__).resolve().parent)
    ap.add_argument('--out',type=Path,default=None)
    actions=ap.add_mutually_exclusive_group(required=True)
    actions.add_argument('--inspect-inputs',action='store_true')
    actions.add_argument('--prepare-only',action='store_true')
    actions.add_argument('--run-kraken',action='store_true')
    actions.add_argument('--assemble-only',action='store_true')
    actions.add_argument('--export-min',action='store_true')
    ap.add_argument('--kraken-exe',type=Path)
    ap.add_argument('--limit',type=int,help='Run only first N cases for parser/solver smoke-check')
    args=ap.parse_args()
    project=args.project_root.resolve();out=(args.out or project/'outputs'/'savex15_stageA_physics_repaired_v1').resolve()
    if args.inspect_inputs:inspect(project)
    elif args.prepare_only:prepare(project,out)
    elif args.assemble_only:assemble(out)
    elif args.export_min:export_min(out)
    else:
        if not args.kraken_exe:ap.error('--run-kraken requires --kraken-exe')
        run_kraken(out,args.kraken_exe.resolve(),limit=args.limit)


if __name__=='__main__': main()
