import sys,json
from pathlib import Path
import torch
ROOT=Path('/data/autovla/projects')
sys.path.insert(0,str(ROOT/'remogen_official_release'))
from scripts.export_reaction100_official_mixed50 import prepare_dataset,run_one,official,Mixed50MLDArgs
out=ROOT/'MoReAct/outputs/remogen_generation_comparison'
items=[]
for name in ['hug','handshake']:
 m=json.loads((ROOT/f'MoReAct/outputs/stage2_best_gifs/{name}/metadata.json').read_text());e=m['episode']
 items.append(dict(record_id=e['episode']+'_a'+e['actor_id']+'_r'+e['reactor_id'],episode=e['episode'],actor_person=e['actor_id'],reactor_person=e['reactor_id'],target_text=m['text'],category=name))
(out/'selection.json').write_text(json.dumps(items,indent=2))
prepare_dataset(items,ROOT/'ttr_remogen_bridge/data/interx_smplx_full_10fps',out/'input')
official.MLDArgs=Mixed50MLDArgs
torch.set_num_threads(2)
run_one('official_text',ROOT/'remogen_official_release/checkpoints/interx_hhi_adapter/checkpoint.pt',out/'input',out,items,torch.device('cuda:6'))
