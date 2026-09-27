import sys,json,shutil
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
from moreact.generate import rollout
from moreact.train import load_checkpoint
out=Path('outputs/best_compare_20260927'); run=Path('runs/diffusion_stage2_b512_stratified_20260926');summary={}
for tag,file in [('best','best.pt'),('best_rollout','best_rollout.pt')]:
 dst=out/(tag+'.pt');shutil.copy2(run/file,dst);s=load_checkpoint(dst);print(tag,s['step'],flush=True)
 summary[tag]={'step':s['step'],'source':str(run/file),'samples':{}}
 for name in ['hug','handshake']:
  m=json.loads(Path(f'outputs/stage2_best_gifs/{name}/metadata.json').read_text())
  rollout(str(dst),str(out/tag/name),episode=m['episode']['episode'],frames=m['frames'],device='cuda:5',seed=m['seed'],guidance=m['guidance'],text=m['text'],video=False,start_frame=m['start_frame'])
  meta=json.loads((out/tag/name/'metadata.json').read_text());summary[tag]['samples'][name]=meta
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
