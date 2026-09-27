"""Paired Inter-X reconstruction; official pretrained VAE transfers to 30 FPS."""
import json
import pickle
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OFFICIAL = ROOT.parent / 'remogen_official_release'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(OFFICIAL))
from model.mld_vae import AutoMldVae
from moreact.models import ReactionVAE
from moreact.train import load_checkpoint
from moreact.data import condition_window
from moreact.geometry import BodyModels, JOINTS, DTRANS, DROT, DJOINTS, reference_frame, transform_features
from moreact.evaluate import motion_metrics, aggregate
from moreact.config import dump_json


@torch.no_grad()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(0)
    device = 'cuda:2'
    saved = load_checkpoint(ROOT / 'runs/cvae_h2_f8_equal_roles_20260924_163259/interim_eval_30000/checkpoint.pt')
    cfg = saved['config']
    ours = ReactionVAE(cfg).to(device).eval()
    ours.load_state_dict(saved['ema'])
    other = AutoMldVae(nfeats=276, latent_dim=[1,256], h_dim=256, ff_size=1024,
                       num_layers=7, num_heads=4).to(device).eval()
    weights = torch.load(OFFICIAL / 'checkpoints/mvae_hml3d/checkpoint.pt', map_location='cpu')
    other.load_state_dict(weights['model_state_dict'])
    with (OFFICIAL / 'assets/mean_std_h2_f8.pkl').open('rb') as f:
        mean, std = pickle.load(f)
    mean, std = torch.as_tensor(mean, device=device), torch.as_tensor(std, device=device)
    stats = {k:v.to(device) for k,v in saved['stats'].items()}
    body = BodyModels(cfg['data']['body_models'], device)
    cache = Path(cfg['data']['cache'])
    meta = json.loads((cache / 'manifest.json').read_text())
    assert meta['digest'] == saved['data_digest']
    records = [r for r in meta['records'] if r['split']=='val'][:32]
    groups = {'MoReAct':[], 'ReMoGen_official':[]}
    for record in records:
        with np.load(cache / record['file']) as z:
            x = torch.tensor(z['features'],device=device)
            offsets = torch.tensor(z['offsets'],device=device)
            betas = torch.tensor(z['betas'],device=device)
            genders = torch.tensor(z['genders'],device=device)
        # Extra next frame for official forward deltas; score only complete blocks.
        frames = min(120, (x.shape[1]-3)//8*8)
        if frames < 8:
            raise ValueError('Evaluation episode too short: '+record['episode'])
        forward = x[1:2,:-1].clone()
        for channel in (DTRANS,DROT,DJOINTS):
            forward[...,channel] = x[1:2,1:,channel]
        predictions = {k:[] for k in groups}
        for end in range(2,2+frames,8):
            ah=x[0:1,end-2:end];rh=x[1:2,end-2:end]
            target=x[1:2,end:end+8]
            a,r,origin,basis=condition_window(ah,rh,offsets[None],stats)
            truth=(transform_features(target,origin,basis,offsets[1:2])-stats['mean'])/stats['std']
            _,mu,_=ours.encode(a,r,truth)
            pred=ours.decode(mu,a,r)*stats['std']+stats['mean']
            world=transform_features(pred,origin,basis,offsets[1:2],inverse=True)
            predictions['MoReAct'].append(body.repair(world,rh,betas[1:2],genders[1:2])[0].cpu().numpy())
            # Official canonicalization anchors the FIRST history frame pelvis in 3D.
            origin,basis=reference_frame(rh[:,:1])
            origin=rh[:,0,JOINTS].reshape(1,22,3)[:,0]
            local=transform_features(forward[:,end-2:end+8],origin,basis,offsets[1:2])
            normalized=(local-mean)/std
            _,dist=other.encode(normalized[:,2:],normalized[:,:2])
            pred=other.decode(dist.mean,normalized[:,:2],8)*std+mean
            world=transform_features(pred,origin,basis,offsets[1:2],inverse=True)
            predictions['ReMoGen_official'].append(body.repair(world,rh,betas[1:2],genders[1:2])[0].cpu().numpy())
        for name in groups:
            pred=np.concatenate(predictions[name])
            metrics=motion_metrics(pred,x[1,2:2+frames].cpu().numpy(),x[0,2:2+frames].cpu().numpy(),30,8,x[1,1].cpu().numpy())
            groups[name].append({'episode':record['episode'],'frames':frames,'metrics':metrics})
        print(record['episode'], flush=True)
    result={'fps':30,'history':2,'future':8,'split':'val','episodes':32,
            'checkpoints':{'MoReAct':saved['step'],'ReMoGen_official':weights['num_steps']},
            'notes':['Posterior-mean reconstruction with true history and true future for both models.',
                     'Same frames, actual genders/betas, common FK repair and physical-space metrics.',
                     'Official model trained on HumanML3D at 10 FPS with neutral gender/zero betas: this is a 30 FPS Inter-X transfer test.',
                     'Official retains forward-difference future lookahead; MoReAct uses causal backward differences.',
                     'This does not isolate architecture, training data, convergence or causal generation quality.'],
            'groups':{name:{'episodes':items,'aggregate':aggregate(items)} for name,items in groups.items()}}
    dump_json(ROOT / 'outputs/remogen_vae_comparison/reconstruction.json',result)


if __name__=='__main__':
    main()
