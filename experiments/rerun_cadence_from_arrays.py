"""Frame-index-only offline reset audit; both inverse and reference use i9606."""
from pathlib import Path
import sys,csv,json,time,importlib.util
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import dynamic_local_basis_coupling_suite as DYN
from mode_fri.causal_ssp_modal_tracking import CSMTConfig,CausalSSPModalTracking
EXP=ROOT/'SWellEx96_S5_CSMT';OUT=EXP/'results/fixed_cadence_reinitialization_20261018'
def csvread(p):
 if not p.exists():return []
 with p.open(encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
 OUT.mkdir(parents=True,exist_ok=True);d=np.load(EXP/'processed/Y_frames.npz');Y=d['Y'];raw=d['receiver_depth_m'];recv=np.round(raw/.125)*.125
 case=EXP/'results/49hz_kraken_reference_stage2/cases/P1_const_base/P1_const_base'
 points=[]
 for l in case.with_suffix('.env').read_text().splitlines()[5:]:
  f=l.split('/') [0].split()
  if len(f)<2:continue
  try:z,c=float(f[0]),float(f[1])
  except ValueError:continue
  points.append((z,c))
  if abs(z-216.5)<1e-8:break
 points=np.array(points);ssp=np.interp(np.arange(0,216.5+.0625,.125),points[:,0],points[:,1])
 cfgfull=dict(grid_size=512,top_n_hypotheses=5,beam_branch_factor=6,beam_width=24,grid_min_separation_rad_m=.00066738)
 full={int(r['frame']):r for r in csvread(OUT/'full_estimates.csv')}
 if 2 not in full:
  r=next(r for r in csvread(EXP/'results/nnfr_time_matched_ssp/time_matched_ssp_nnfr.csv') if int(r['frame_index'])==2 and r['status']=='OK')
  full[2]=dict(frame=2,k1=float(r['k1_time_matched']),k2=float(r['k2_time_matched']),k3=float(r['k3_time_matched']),runtime=float(r['runtime_seconds']),accepted=r['refinement_accepted'])
 for frame in range(2,68,5):
  if frame in full:continue
  print('Full frame',frame,flush=True);t0=time.perf_counter()
  meta=DYN.DynamicMetadata(K=3,frequency=49,array_geometry=recv.copy(),water_depth=216.5,SSP=ssp.copy())
  k,_,accepted,_,_=DYN.dynamic_topn_refined(Y=Y[frame],metadata=meta,seed=20260917+1000*frame+1,cfg=cfgfull,dz=.125)
  k=np.sort(k)[::-1];full[frame]=dict(frame=frame,k1=float(k[0]),k2=float(k[1]),k3=float(k[2]),runtime=time.perf_counter()-t0,accepted=bool(accepted))
  csvwrite(OUT/'full_estimates.csv',[full[f] for f in sorted(full)]);print(full[frame],flush=True)
 spec=importlib.util.spec_from_file_location('kref',ROOT/'NN_FRI_49Hz_KRAKEN_reference_stage2.py');m=importlib.util.module_from_spec(spec);sys.modules['kref']=m;spec.loader.exec_module(m)
 roots=m.parse_mod(case.with_suffix('.mod'),raw)[0].real
 def kval(r):return np.array([float(r['k1']),float(r['k2']),float(r['k3'])])
 def assignment(k):
  ri,ci=linear_sum_assignment(abs(k[:,None]-roots[None,:]));a=np.empty(3,int);a[ri]=ci;return a
 initial=assignment(kval(full[2]));scale=.002169/.013
 cfg=CSMTConfig(delta0_rad_m=.0012*scale,delta_min_rad_m=.0008*scale,delta_max_rad_m=.004*scale,minimum_mode_separation_rad_m=.0035*scale)
 frames=[];summary=[];arrays={}
 for name,period in [('no_reset',None),('period20',20),('period10',10),('period05',5)]:
  tr=CausalSSPModalTracking(cfg);ks=[]
  for frame in range(2,70):
   reset=frame==2 or period is not None and (frame-2)%period==0
   args=dict(SSP_t=ssp,Y_t=Y[frame],frequency_hz=49,dz=.125,receiver_depths=recv,water_depth=216.5)
   res=tr.initialize_from_full_estimate(k_hat=kval(full[frame]),full_runtime_seconds=float(full[frame]['runtime']),**args) if reset else tr.step(**args)
   k=res.k_hat;ks.append(k);nearest=abs(k[:,None]-roots[None,:]).argmin(axis=1);uniq=assignment(k)
   frames.append(dict(schedule=name,frame=frame,fixed_mae=float(np.mean(abs(k-roots[initial]))),unique_mae=float(np.mean(abs(k-roots[uniq]))),retained=int(np.all(nearest==initial)),residual=res.residual_mean,runtime=res.total_runtime_seconds,k1=k[0],k2=k[1],k3=k[2]))
  arrays[name]=np.array(ks);sc=[r for r in frames if r['schedule']==name and r['frame']>2];allr=[r for r in frames if r['schedule']==name]
  summary.append(dict(schedule=name,fixed_mae=float(np.mean([r['fixed_mae'] for r in sc])),unique_mae=float(np.mean([r['unique_mae'] for r in sc])),retained=sum(r['retained'] for r in sc),scored_frames=67,residual=float(np.mean([r['residual'] for r in sc])),runtime=sum(r['runtime'] for r in allr)));print(summary[-1],flush=True)
 csvwrite(OUT/'frame_metrics.csv',frames);csvwrite(OUT/'summary.csv',summary)
 np.savez_compressed(OUT/'trajectories.npz',model_roots=roots,initial_assignment=initial,**arrays)
 (OUT/'RUN_METADATA.json').write_text(json.dumps(dict(trigger='original frame index only',supplied_ssp='i9606 from audited P1_const_base.env',reference='13 KRAKEN roots; i9606',scored_frames='3--69',initial_assignment=(initial+1).tolist(),parameters=cfg.__dict__,Full_config=cfgfull),indent=2))
 fig,axs=plt.subplots(3,1,figsize=(7.3,7.2),sharex=True)
 for j,ax in enumerate(axs):
  ax.axhline(roots[initial[j]],ls='--',c='.6',lw=1,label='initial assigned model root')
  for name,k in arrays.items():ax.plot(np.arange(2,70),k[:,j],lw=1,label=name)
  ax.set_ylabel('$k_'+str(j+1)+'$ (rad/m)');ax.grid(alpha=.2)
 axs[0].legend(ncol=3,fontsize=7);axs[-1].set_xlabel('Original SWellEx frame');fig.tight_layout();fig.savefig(OUT/'Supp_FigS7_fixed_cadence_trajectories.pdf');plt.close(fig)
if __name__=='__main__':main()
