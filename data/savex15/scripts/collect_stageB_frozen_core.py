# -*- coding: utf-8 -*-
"""Collect code and locks ONLY. Never runs experiments or alters the original NN_FRI project.
Run: py collect_stageB_frozen_core.py  [D:\\NN_FRI_Mode_Extension]
"""
from __future__ import annotations
import hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(sys.argv[1] if len(sys.argv)>1 else r'D:\NN_FRI_Mode_Extension')
OUT=ROOT/'NN_FRI_StageB_FROZEN_CORE_MIN.zip'
# Reference hashes from prior archived, independently acquired core snapshot: not proof that current checkout is frozen.
ARCHIVED={
'src/mode_fri/causal_ssp_modal_tracking.py':'69a9d8d7d7da82899fad3cfce7fdc8b20d6ba3a6126d0ef42e0e8703b239497f',
'src/mode_fri/online_track_reacquire.py':'d6acf08bd4b34337c6a953fbda33987f975fe91ab9b766f8b2377cc7de2a18d5',
'src/mode_fri/direct_estimator.py':'8bc384df6bbeb8126f67fcc84bb7f3f2d28964dd49ce46daedd8fc041b177e0a',
'src/mode_fri/mode_frinet.py':'ee0eb6bc544eca37d60d8e5a3f8d24901c00e0c69e67f506bd8f5dd5d0b8da05',
'experiments/csmt_v2_independent_holdout.py':'5fcc8c460f2ba9d2acc4ab533ab58835a8841212e5a1532b077a19d756a1f500',
'outputs/csmt_v2_independent_holdout/CSMT_V2_PARAMETER_LOCK.json':'188dfae9ff4617bf1a60427a011b0118ba23d09cd5a47dd9deff4eb6f8f66034',
}
required=['src/mode_fri/causal_ssp_modal_tracking.py','src/mode_fri/online_track_reacquire.py','src/mode_fri/direct_estimator.py',
          'src/mode_fri/mode_frinet.py','experiments/csmt_v2_independent_holdout.py',
          'outputs/csmt_v2_independent_holdout/CSMT_V2_PARAMETER_LOCK.json']
if not ROOT.is_dir(): raise SystemExit(f'Original NN_FRI project not found: {ROOT}')
files=set()
for pattern in ['src/mode_fri/**/*.py','configs/*.json','configs/*.yaml','configs/*.yml',
                'experiments/csmt_v2_independent_holdout.py',
                'SWellEx96_S5_CSMT/experiments/05_swellex_nnfr_representative_frames.py',
                'outputs/csmt_v2_independent_holdout/*LOCK*.json',
                'outputs/cmtr_ablation_v1/*LOCK*.json']:
    for p in ROOT.glob(pattern):
        if p.is_file() and p.stat().st_size<1024*1024: files.add(p)
files.update(p for k in required if (p:=ROOT/k).is_file())
manifest={'root':str(ROOT),'reference_snapshot_note':'Archival hashes verify only historical code identity, not the current final runtime correctness.',
          'files':{},'required_missing':[], 'archived_hash_comparisons':{}}
for path in required:
    p=ROOT/path
    if not p.is_file():manifest['required_missing'].append(path)
    else:
        got=hashlib.sha256(p.read_bytes()).hexdigest()
        manifest['archived_hash_comparisons'][path]={'actual':got,'archived':ARCHIVED[path], 'matches_archived':got==ARCHIVED[path]}
        print(('MATCH' if got==ARCHIVED[path] else 'DIFF '),path)
for p in sorted(files):
    name=p.relative_to(ROOT).as_posix()
    manifest['files'][name]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
with zipfile.ZipFile(OUT,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=7) as z:
    for p in sorted(files):z.write(p,p.relative_to(ROOT).as_posix())
    z.writestr('FROZEN_CORE_COLLECT_MANIFEST.json',json.dumps(manifest,ensure_ascii=False,sort_keys=True,indent=2))
print('COLLECTED',len(files),'files; MISSING',len(manifest['required_missing']))
print('ZIP:',OUT)
print('No experiments were executed; no original files were modified.')
