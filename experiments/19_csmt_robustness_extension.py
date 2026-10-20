"""Paired update-stage sensitivity; reference initialization isolates local updates."""
from pathlib import Path
import sys,json,csv,hashlib,math
import numpy as np
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mode_fri.causal_ssp_modal_tracking import CSMTConfig,CausalSSPModalTracking
OUT=ROOT/'outputs/csmt_robustness_extension_20261018'
CONDITIONS=[('snr',-5,20,0),('snr',0,20,0),('snr',5,20,0),('nominal',10,20,0),('snr',15,20,0),('snapshots',10,5,0),('snapshots',10,10,0),('snapshots',10,40,0),('shift',10,20,-2),('shift',10,20,-1),('shift',10,20,1),('shift',10,20,2)]
def seed(*x):
 return int.from_bytes(hashlib.sha256('|'.join(['csmt_robustness_extension_v1_20261018',*map(str,x)]).encode()).digest()[:4],'little')
def savecsv(p,rows):
 with p.open('w',newline='',encoding='utf-8-sig') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def observation(b,L,snr,r,t):
 g=np.random.default_rng(seed(r,t,'coeff'))
 c=(g.standard_normal((3,L))+1j*g.standard_normal((3,L)))/math.sqrt(2)
 c*=np.exp(-.22*np.arange(3))[:,None];clean=b@c
 g=np.random.default_rng(seed(r,t,'noise'));n=(g.standard_normal(clean.shape)+1j*g.standard_normal(clean.shape))/math.sqrt(2)
 return clean+n*math.sqrt(np.linalg.norm(clean)**2/max(np.linalg.norm(n)**2*10**(snr/10),1e-30))
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 a=np.load(ROOT/'outputs/kraken_timevarying_csmt_crosscheck_formal/arrays/kraken_dynamic_truth.npz')
 truth=a['truth_k_rad_m'];ssp=a['ssp_m_s'];z=a['depths_m'];b=a['kraken_receiver_basis'];recv=a['receiver_depths_m']
 cfg=CSMTConfig(**json.loads((ROOT/'configs/csmt_v2_independent_holdout.json').read_text())['csmt_v2'])
 ks=np.zeros((12,20,21,3));pred=np.zeros_like(ks);width=np.zeros_like(ks);summ=[];frames=[];reps=[]
 for ci,(factor,snr,L,shift) in enumerate(CONDITIONS):
  print('condition',ci,factor,snr,L,shift,flush=True)
  inv=np.array([np.interp(z-shift,z,p,left=p[0],right=p[-1]) for p in ssp])
  for r in range(20):
   tr=CausalSSPModalTracking(cfg)
   for t in range(21):
    Y=observation(b[t],L,snr,r,t)
    args=dict(SSP_t=inv[t],Y_t=Y,frequency_hz=300,dz=1,receiver_depths=recv,water_depth=40)
    res=tr.initialize_from_full_estimate(k_hat=truth[0],full_runtime_seconds=0,**args) if t==0 else tr.step(**args)
    ks[ci,r,t]=res.k_hat;pred[ci,r,t]=res.predicted_k;width[ci,r,t]=res.search_half_widths
    if t:
     e=res.k_hat-truth[t];outside=abs(truth[t]-res.predicted_k)>res.search_half_widths
     frames.append(dict(condition=ci,repeat=r,frame=t,rmse=float(np.sqrt(np.mean(e**2))),success=int(max(abs(e))<=.0015),outside=float(np.mean(outside))))
   e=ks[ci,r,1:]-truth[1:]
   reps.append(dict(condition=ci,repeat=r,rmse=float(np.sqrt(np.mean(e**2)))))
  e=ks[ci,:,1:]-truth[None,1:];rv=np.sqrt(np.mean(e**2,axis=(1,2)))
  rng=np.random.default_rng(20261018);means=rv[rng.integers(0,20,(20000,20))].mean(axis=1);lo,hi=np.quantile(means,[.025,.975])
  summ.append(dict(condition=ci,factor=factor,snr_db=snr,snapshots=L,shift_m=shift,rmse=float(np.sqrt(np.mean(e**2))),ci_low=float(lo),ci_high=float(hi),success=float(np.mean(np.max(abs(e),axis=2)<=.0015)),outside=float(np.mean(abs(truth[None,1:]-pred[ci,:,1:])>width[ci,:,1:]))))
 np.savez_compressed(OUT/'trajectories.npz',csmt_k=ks,predicted_k=pred,search_half_widths=width,truth_k=truth)
 savecsv(OUT/'summary.csv',summ);savecsv(OUT/'frame_metrics.csv',frames);savecsv(OUT/'repeat_metrics.csv',reps)
 (OUT/'RUN_METADATA.json').write_text(json.dumps(dict(seed_namespace='csmt_robustness_extension_v1_20261018',conditions=CONDITIONS,initialization='frame-0 KRAKEN reference roots; update-stage test',shift_definition='inverse(z)=forward(z-shift); endpoint continuation',point_metric='pooled update-component RMSE',interval='bootstrap mean of 20 repeat RMSEs',parameters=cfg.__dict__),indent=2))
 fig,axs=plt.subplots(1,3,figsize=(10.8,3.5))
 for ax,factor,xkey,xlab in zip(axs,['snr','snapshots','shift'],['snr_db','snapshots','shift_m'],['SNR (dB)','Snapshots per frame','Inverse-SSP depth shift (m)']):
  rows=sorted([s for s in summ if s['factor'] in [factor,'nominal']],key=lambda s:s[xkey]);x=[s[xkey] for s in rows];y=np.array([s['rmse'] for s in rows])*1e3
  ax.errorbar(x,y,yerr=np.array([[s['rmse']-s['ci_low'] for s in rows],[s['ci_high']-s['rmse'] for s in rows]])*1e3,marker='o',capsize=3);ax.set_xlabel(xlab);ax.set_ylabel('Tracking RMSE ($10^{-3}$ rad/m)');ax.grid(alpha=.25)
 fig.tight_layout();fig.savefig(OUT/'Supp_FigS6_robustness_sweeps.pdf');plt.close(fig)
 print(json.dumps(summ,indent=2),flush=True)
if __name__=='__main__':main()
