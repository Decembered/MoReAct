import sys,json,pickle,importlib.util
from pathlib import Path
import numpy as np,torch
sys.path.insert(0,str(Path.cwd()))
from moreact.generate import ReactionGenerator,rollout_arrays
from moreact.geometry import JOINTS
spec=importlib.util.spec_from_file_location('reference','docs/reports/20260927_three_model_sample/evaluate.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
out=Path('outputs/best_compare_20260927');selection=[x for x in json.loads((ref.OUT/'selection.json').read_text()) if x['split']=='test'];rows=[]
models={'old8750':Path('outputs/stage2_best_gifs/checkpoint.pt'),'best':out/'best.pt','best_rollout':out/'best_rollout.pt'}
for tag,ckpt in models.items():
 gen=ReactionGenerator(ckpt,'cuda:5',seed=0,guidance=1.)
 for item in selection:
  with np.load(ref.CACHE/item['file']) as z:data={k:z[k] for k in z.files}
  gen.rng.manual_seed(0)
  pred,_=rollout_arrays(gen,torch.from_numpy(data['features'][0,:122].copy()),torch.from_numpy(data['features'][1,:2].copy()),torch.from_numpy(data['betas']),torch.from_numpy(data['genders']),torch.from_numpy(data['offsets']),120,torch.from_numpy(data['text_embeddings'][:1].copy()))
  p=pred.numpy()[:,JOINTS].reshape(-1,22,3);q=data['features'][1,2:122,JOINTS].reshape(-1,22,3);a=data['features'][0,2:122,JOINTS].reshape(-1,22,3)
  rows.append(dict(model=tag,episode=item['episode'],metrics=ref.metrics(p,q,a)))
 print(tag,'20 done',flush=True)
 del gen;torch.cuda.empty_cache()
for item in selection:
 r=pickle.loads((ref.OUT/'official/text/seed_0'/(item['record_id']+'.pkl')).read_bytes());rows.append(dict(model='ReMoGen',episode=item['episode'],metrics=ref.metrics(*[r[k]['joints'][2:122] for k in ['reactor','gt_reactor','actor']])))
summary={tag:{k:float(np.mean([r['metrics'][k] for r in rows if r['model']==tag])) for k in rows[0]['metrics']} for tag in [*models,'ReMoGen']}
rng=np.random.default_rng(0);paired={};old=[r for r in rows if r['model']=='old8750']
for tag in ['best','best_rollout']:
 new=[r for r in rows if r['model']==tag];paired[tag]={}
 for k in summary[tag]:
  d=np.array([a['metrics'][k]-b['metrics'][k] for a,b in zip(new,old)]);paired[tag][k]={'mean_delta':float(d.mean()),'wins':int((d<0).sum()),'bootstrap95':np.percentile(d[rng.integers(0,20,(10000,20))].mean(1),[2.5,97.5]).tolist()}
(out/'test20.json').write_text(json.dumps(dict(summary=summary,paired=paired,rows=rows,selection=selection),indent=2));print(json.dumps(summary),flush=True)
