"""Frozen-forward, paired CSMT SSP perturbation audit (20x21 per condition).

Uniform offset: c_inv(z,t) = c_fwd(z,t) + delta [m/s].
Contrast: c_inv(z,t)=c_fwd(z,t)+delta*(c_fwd(z,t)-c_mid(t))/c_half(t),
  where c_mid=(max(c)+min(c))/2 and c_half=(max(c)-min(c))/2.
Thus delta is the magnitude of the top/bottom speed perturbation (m/s);
the full top-to-bottom contrast changes by 2*delta m/s. This is a
fixed-forward, inversion-only sensitivity, NOT a refit of physical data.
Frozen ahead of results: offsets ±1/±2, contrasts ±1/±2 and one nominal.
Same paired observations, frame-0 KRAKEN-reference initialization and
unchanged CSMTConfig as experiment 19; successful if all 3 |errors| ≤0.0015.
"""
import csv, json, hashlib, importlib.util, sys, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
CORE=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else HERE.parent
ORIGINAL=CORE/'experiments'/'19_csmt_robustness_extension.py'
spec=importlib.util.spec_from_file_location('robust_ext_orig',ORIGINAL)
orig=importlib.util.module_from_spec(spec);spec.loader.exec_module(orig)
from mode_fri.causal_ssp_modal_tracking import CSMTConfig, CausalSSPModalTracking
OUT=CORE/'outputs'/'csmt_ssp_profile_perturbation_20261018'
# Prespecified test conditions; result-independent ordering and amplitudes.
CONDITIONS=[('nominal',0.0),('offset',-2.0),('offset',-1.0),('offset',1.0),('offset',2.0),('contrast',-2.0),('contrast',-1.0),('contrast',1.0),('contrast',2.0)]
REPEATS=20;FRAMES=21;K=3;SNR_DB=10;L=20
SEED_NS='csmt_robustness_extension_v1_20261018'
METRIC_TOL=0.0015

def inv_profiles(ssp,kind,delta):
  if kind=='nominal': return ssp.copy()
  if kind=='offset': return ssp+delta
  if kind=='contrast':
    top=ssp.max(axis=1,keepdims=True);bot=ssp.min(axis=1,keepdims=True)
    mid=.5*(top+bot);half=.5*(top-bot)
    if np.any(half<1.e-12): raise ValueError('zero contrast')
    return ssp+delta*(ssp-mid)/half
  raise ValueError(kind)

def csvsave(dest,rows):
  with dest.open('w',newline='',encoding='utf-8') as f:
    wr=csv.DictWriter(f,fieldnames=list(rows[0]));wr.writeheader();wr.writerows(rows)

def main():
  OUT.mkdir(exist_ok=True,parents=True)
  a=np.load(CORE/'outputs'/'kraken_timevarying_csmt_crosscheck_formal'/'arrays'/'kraken_dynamic_truth.npz')
  truth=a['truth_k_rad_m'];ssp=a['ssp_m_s'];z=a['depths_m'];basis=a['kraken_receiver_basis'];recv=a['receiver_depths_m']
  assert truth.shape==(FRAMES,K) and ssp.shape[0]==FRAMES and basis.shape[0]==FRAMES
  cfg=CSMTConfig(**json.loads((CORE/'configs/csmt_v2_independent_holdout.json').read_text())['csmt_v2'])
  k_arr=np.full((len(CONDITIONS),REPEATS,FRAMES,K),np.nan)
  pred_arr=np.full_like(k_arr,np.nan);width_arr=np.full_like(k_arr,np.nan)
  summary=[];reps=[];frames=[]
  t_begin=time.perf_counter()
  for ci,(kind,delta) in enumerate(CONDITIONS):
    inv=inv_profiles(ssp,kind,delta)
    for r in range(REPEATS):
      tr=CausalSSPModalTracking(cfg)
      for t in range(FRAMES):
        Y=orig.observation(basis[t],L,SNR_DB,r,t)
        args=dict(SSP_t=inv[t],Y_t=Y,frequency_hz=300,dz=1,receiver_depths=recv,water_depth=40)
        res=(tr.initialize_from_full_estimate(k_hat=truth[0],full_runtime_seconds=0,**args) if t==0 else tr.step(**args))
        k_arr[ci,r,t]=res.k_hat;pred_arr[ci,r,t]=res.predicted_k;width_arr[ci,r,t]=res.search_half_widths
        if t:
          e=res.k_hat-truth[t]
          frames.append(dict(condition=ci,perturbation=kind,delta_m_s=delta,repeat=r,frame=t,rmse=float(np.sqrt(np.mean(e**2))),success=int(np.max(np.abs(e))<=METRIC_TOL)))
      e=k_arr[ci,r,1:]-truth[1:]
      reps.append(dict(condition=ci,perturbation=kind,delta_m_s=delta,repeat=r,rmse=float(np.sqrt(np.mean(e**2))),success_rate=float(np.mean(np.max(np.abs(e),axis=1)<=METRIC_TOL))))
    err=k_arr[ci,:,1:]-truth[None,1:]
    per_rep=np.sqrt(np.mean(err**2,axis=(1,2)))
    rng=np.random.default_rng(20261018)
    choices=rng.integers(0,REPEATS,size=(20000,REPEATS))
    means=per_rep[choices].mean(axis=1)
    # pooled update-component RMSE comparable directly to original experiment 19
    pooled=float(np.sqrt(np.mean(err**2)))
    success=float(np.mean(np.max(np.abs(err),axis=2)<=METRIC_TOL))
    summary.append(dict(condition=ci,perturbation=kind,delta_m_s=delta,snr_db=SNR_DB,snapshots=L,
      rmse_pooled=pooled,mean_repeat_rmse=float(np.mean(per_rep)),median_repeat_rmse=float(np.median(per_rep)),bootstrap_mean_lo=float(np.quantile(means,.025)),bootstrap_mean_hi=float(np.quantile(means,.975)),success=success,
      ratio_to_original_nominal='',frames=REPEATS*(FRAMES-1)))
    print(f'{ci+1}/{len(CONDITIONS)} {kind} {delta:+g}: pooled={pooled:.8g} success={success:.4f} elapsed={time.perf_counter()-t_begin:.1f}s',flush=True)
  original_rows=list(csv.DictReader((CORE/'outputs/csmt_robustness_extension_20261018/summary.csv').open(encoding='utf-8-sig')))
  original_nominal=next(row for row in original_rows if row['factor']=='nominal')
  orig_nominal=float(original_nominal['rmse']);orig_succ=float(original_nominal['success'])
  np.testing.assert_allclose(summary[0]['rmse_pooled'],orig_nominal,rtol=0,atol=2.e-15)
  np.testing.assert_allclose(summary[0]['success'],orig_succ,rtol=0,atol=0)
  old=np.load(CORE/'outputs/csmt_robustness_extension_20261018/trajectories.npz')
  np.testing.assert_array_equal(k_arr[0],old['csmt_k'][3])
  for row in summary:row['ratio_to_original_nominal']=row['rmse_pooled']/orig_nominal
  csvsave(OUT/'summary.csv',summary);csvsave(OUT/'repeat_metrics.csv',reps);csvsave(OUT/'frame_metrics.csv',frames)
  np.savez_compressed(OUT/'trajectories.npz',csmt_k=k_arr,predicted_k=pred_arr,search_half_widths=width_arr,truth_k=truth,original_forward_ssp_m_s=ssp)
  fig,axs=plt.subplots(1,2,figsize=(9.8,3.5),sharey=False)
  for ax,kind in zip(axs,['offset','contrast']):
    rows=sorted([x for x in summary if x['perturbation'] in ('nominal',kind)],key=lambda x:x['delta_m_s'])
    ax.plot([x['delta_m_s'] for x in rows],[x['rmse_pooled']*1e3 for x in rows],marker='o',label='Pooled update RMSE')
    ax.set_title('Uniform speed offset' if kind=='offset' else 'Thermocline contrast variation')
    ax.set_xlabel('Input SSP perturbation (m/s)');ax.set_ylabel('Tracking RMSE ($10^{-3}$ rad/m)');ax.grid(alpha=.25)
  fig.tight_layout();fig.savefig(OUT/'SSP_profile_perturbation.pdf');plt.close(fig)
  fingerprints={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ORIGINAL,CORE/'configs/csmt_v2_independent_holdout.json',CORE/'outputs/kraken_timevarying_csmt_crosscheck_formal/arrays/kraken_dynamic_truth.npz']}
  metadata=dict(protocol='Frozen forward observations; inverse input SSP only',conditions=CONDITIONS,confidence_intervals='20000 repeat-level paired-bootstrap draws; bootstrap mean of repeat RMSE, NOT pooled RMSE',
    seed_namespace=SEED_NS,reference_init='frame-0 model roots, same as experiment 19',repeats=REPEATS,frames=FRAMES,snr_db=SNR_DB,snapshots=L,mode_count=K,frequency_hz=300,water_depth_m=40,
    success_rule='max(abs(k_hat-truth)) <= 0.0015 rad/m',perturbation_model='offset: + delta m/s all depths; contrast: + delta*(c-mid)/half, top +delta bottom -delta m/s',
    forward_basis_unchanged=True,inverse_params_not_retuned=True,initialization_failure_not_tested=True,independent_environments_not_tested=True,
    input_sha256=fingerprints,original_nominal_rmse=orig_nominal,original_nominal_success=orig_succ,
    elapsed_seconds=time.perf_counter()-t_begin)
  (OUT/'PROTOCOL_AND_QA.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
  (OUT/'INDEPENDENT_RECOMPUTE.txt').write_text('PASS: New nominal reproduces experiment 19 original nominal RMSE, success and all estimated frame trajectories exactly.\n',encoding='utf-8')
  print('ALL NEW EXPERIMENT AND NOMINAL CROSS-CHECKS PASSED',flush=True)

if __name__=='__main__':main()
