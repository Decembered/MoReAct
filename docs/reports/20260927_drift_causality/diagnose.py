"""Frozen-checkpoint interventions; no training parameter or optimizer updates."""
import json
import math
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path('/data/autovla/projects/MoReAct')
sys.path.insert(0, str(ROOT))
from moreact.generate import ReactionGenerator
from moreact.data import condition_window
from moreact.geometry import JOINTS, TRANSL, POSE, transform_features, rotation_6d_to_matrix
from moreact.losses import diffusion_motion_losses, huber

OUT = ROOT/'outputs/drift_causality_20260927'
CACHE = ROOT/'data/interx_h2_f8'
MODELS = {
    'root30': ROOT/'runs/diffusion_scratch_b768_root30_20260926/step_008500.pt',
    'baseline': ROOT/'runs/diffusion_stage2_b512_stratified_20260926/step_021000.pt',
}


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/name).write_text(json.dumps(value, indent=2)+'\n')


def yaw_matrix(angle, device):
    c,s = math.cos(angle), math.sin(angle)
    return torch.tensor([[c,-s,0],[s,c,0],[0,0,1]],device=device,dtype=torch.float32)[None]


def yaw(features):
    rot = rotation_6d_to_matrix(features[...,POSE].reshape(*features.shape[:-1],22,6))[...,0,:,:]
    return torch.atan2(rot[...,1,0],rot[...,0,0])


def wrap(angle):
    return torch.atan2(torch.sin(angle),torch.cos(angle))


def move(history, offsets, shift=(0.,0.,0.), angle=0.):
    pivot = history[:,-1,JOINTS].reshape(-1,22,3)[:,0]
    rot = yaw_matrix(angle,history.device)
    origin = pivot-torch.einsum('bij,bj->bi',rot,pivot)+history.new_tensor(shift)[None]
    return transform_features(history,origin,rot,offsets,inverse=True)


def predict(g, ah, rh, data, seed):
    g.rng.manual_seed(seed)
    state = g.initialize(rh,data['betas'][None],data['genders'][None],data['offsets'][None])
    return g.step(ah,state,data['text_embeddings'][:1])[0]


def score(pred,rh,gt,gh):
    p,q = pred[0,:,TRANSL], gt[0,:,TRANSL]
    prev,trueprev = rh[0,-1,TRANSL],gh[0,-1,TRANSL]
    err = p-q
    v = torch.diff(torch.cat((prev[None],p)),dim=0)
    vgt = torch.diff(torch.cat((trueprev[None],q)),dim=0)
    e = prev-trueprev
    correction = -((v-vgt)*e).sum(-1)/(e.norm()+1e-9)
    return dict(root_cm=float(err.norm(dim=-1).mean()*100), first_cm=float(err[0].norm()*100),
        last_cm=float(err[-1].norm()*100), prior_drift_cm=float(e.norm()*100),
        boundary_step_cm=float(v[0].norm()*100), interior_step_cm=float(v[1:].norm(dim=-1).mean()*100),
        boundary_velocity_error_cm=float((v[0]-vgt[0]).norm()*100),
        boundary_accel_cm=float((v[0]-(rh[0,-1,TRANSL]-rh[0,-2,TRANSL])).norm()*100),
        first_correction_cm=float(correction[0]*100),
        heading_deg=float(wrap(yaw(pred)-yaw(gt)).abs().mean()*180/math.pi))


def gradients(g,ah,rh,gh,gt,actor_future,data,seed,timestep):
    offsets=data['offsets'][None]
    a,r,origin,basis=condition_window(ah,rh,offsets,g.stats)
    local=transform_features(gt,origin,basis,offsets[:,1])
    target=(local-g.stats['mean'])/g.stats['std']
    with torch.no_grad():
        torch.manual_seed(seed)
        z,_,_=g.vae.encode(a,r,target)
        z=z/g.vae.latent_scale
        t=torch.tensor([timestep],device=g.device)
        noisy=g.diffusion.corrupt(z,t,torch.randn_like(z))
        estimate=g.model(noisy,t,a,r,data['text_embeddings'][:1])
    estimate=estimate.detach().requires_grad_(True)
    pred=g.vae.decode(estimate*g.vae.latent_scale,a,r)
    weights=dict(g.cfg['diffusion_loss']);weights['root_position']=30.
    _,terms=diffusion_motion_losses(pred,target,r,g.stats['mean'],g.stats['std'],g.body,
        data['betas'][None,1],data['genders'][None,1],weights,
        transform_features(actor_future,origin,basis,offsets[:,0]))
    terms['latent_mse']=F.mse_loss(estimate,z)
    physical=pred*g.stats['std']+g.stats['mean']
    history_local=transform_features(gh,origin,basis,offsets[:,1])
    pseq=torch.cat(((rh.new_tensor(0)+r[:,-1:])*g.stats['std']+g.stats['mean'],physical),1)
    qseq=torch.cat((history_local[:,-1:],local),1)
    terms['true_gt_velocity']=huber(torch.diff(pseq[...,JOINTS],dim=1),torch.diff(qseq[...,JOINTS],dim=1))
    terms['relative_displacement']=F.mse_loss(physical[...,TRANSL]-pseq[:,:1,TRANSL],
                                             local[...,TRANSL]-history_local[:,-1:,TRANSL])
    selected=['root_position','joint_velocity','true_gt_velocity','relative_displacement',
              'feature_rec','foot_contact','latent_mse','root_angular_velocity']
    grads={k:torch.autograd.grad(terms[k],estimate,retain_graph=True)[0].flatten() for k in selected}
    root=grads['root_position']
    result={}
    for k in selected:
        grad=grads[k]
        weight=weights.get(k,1.)
        result[k]=dict(loss=float(terms[k]),weighted_grad_norm=float(grad.norm()*weight),
                       cosine_root=float(F.cosine_similarity(root[None],grad[None],eps=1e-12)))
    return result


def witness(g,ah,gh,gt,data):
    rh=move(gh,data['offsets'][None,1],(.3,0,0))
    a,r,origin,basis=condition_window(ah,rh,data['offsets'][None],g.stats)
    target_local=transform_features(gt,origin,basis,data['offsets'][None,1])
    target=(target_local-g.stats['mean'])/g.stats['std']
    candidates={'snap_to_gt':gt,'preserve_offset':move(gt,data['offsets'][None,1],(.3,0,0))}
    result={}
    prev=(r[:,-1:]*g.stats['std']+g.stats['mean'])[...,JOINTS]
    trueprev=transform_features(gh,origin,basis,data['offsets'][None,1])[:,-1:,JOINTS]
    q=target_local[...,JOINTS]
    for name,world in candidates.items():
        physical=transform_features(world,origin,basis,data['offsets'][None,1])
        p=physical[...,JOINTS]
        result[name]=dict(root30=float(30*F.mse_loss(physical[...,TRANSL],target_local[...,TRANSL])),
            current_velocity100=float(100*huber(torch.diff(torch.cat((prev,p),1),dim=1),torch.diff(torch.cat((prev,q),1),dim=1))),
            true_velocity100=float(100*huber(torch.diff(torch.cat((prev,p),1),dim=1),torch.diff(torch.cat((trueprev,q),1),dim=1))))
    return result


def run():
    selection=json.loads((ROOT/'outputs/three_model_sample_20260927/selection.json').read_text())
    selection=sum(([x for x in selection if x['split']==s][:6] for s in ('train','test')),[])
    save('protocol.json',dict(selection=selection,models={k:str(v) for k,v in MODELS.items()},
        seeds=[0,1,2],injection_cutoff=42,probe_cutoffs=[42,82],frames=120,
        gradients='EMA denoiser output-latent gradients, timesteps 0/5/9, no parameter updates'))
    injections=[];gradrows=[];probes=[];rollouts=[];witnesses=[]
    cases=[('clean',(0,0,0),0),('x+10',(.1,0,0),0),('x-10',(-.1,0,0),0),
           ('x+30',(.3,0,0),0),('x-30',(-.3,0,0),0),('y+30',(0,.3,0),0),
           ('y-30',(0,-.3,0),0),('yaw+20',(0,0,0),math.radians(20)),
           ('yaw-20',(0,0,0),math.radians(-20)),('common_shift',(.3,0,0),0)]
    for model,path in MODELS.items():
        g=ReactionGenerator(path,'cuda:0')
        for item in selection:
            with np.load(CACHE/item['file']) as z:
                data={k:torch.from_numpy(z[k].copy()).to(g.device) for k in
                      ('features','betas','genders','offsets','text_embeddings')}
            feat=data['features'];ah=feat[0,40:42][None];gh=feat[1,40:42][None];gt=feat[1,42:50][None]
            base=dict(model=model,episode=item['episode'],split=item['split'])
            with torch.no_grad():
                witnesses.append(dict(**base,values=witness(g,ah,gh,gt,data)))
                for seed in (0,1,2):
                    clean=None
                    for name,shift,angle in cases:
                        rh=move(gh,data['offsets'][None,1],shift,angle)
                        actor=move(ah,data['offsets'][None,0],shift) if name=='common_shift' else ah
                        target=move(gt,data['offsets'][None,1],shift) if name=='common_shift' else gt
                        reference=move(gh,data['offsets'][None,1],shift) if name=='common_shift' else gh
                        pred=predict(g,actor,rh,data,seed)
                        if name=='clean':clean=pred.clone()
                        metrics=score(pred,rh,target,reference)
                        if name=='common_shift':
                            expected=move(clean,data['offsets'][None,1],shift)
                            metrics['equivariance_max_m']=float((pred[...,TRANSL]-expected[...,TRANSL]).abs().max())
                        else:
                            delta=pred-clean
                            if shift!=(0,0,0):
                                offset=pred.new_tensor(shift);unit=offset/offset.norm()
                                response=(delta[0,:,TRANSL]*unit).sum(-1)/offset.norm()
                                metrics['offset_retained_first']=float(response[0])
                                metrics['offset_retained_last']=float(response[-1])
                        injections.append(dict(**base,seed=seed,case=name,metrics=metrics))
            for name,shift in [('clean',(0,0,0)),('x+30',(.3,0,0)),('x-30',(-.3,0,0))]:
                rh=move(gh,data['offsets'][None,1],shift)
                for timestep in (0,5,9):
                    gradrows.append(dict(**base,case=name,timestep=timestep,
                        terms=gradients(g,ah,rh,gh,gt,feat[0,42:50][None],data,0,timestep)))
            with torch.no_grad():
                for seed in (0,1,2):
                    rh=feat[1,:2][None].clone();generated=[]
                    for cutoff in range(2,122,8):
                        actor=feat[0,cutoff-2:cutoff][None]
                        gt_history=feat[1,cutoff-2:cutoff][None]
                        target=feat[1,cutoff:cutoff+8][None]
                        # Separate deterministic noise per segment, paired across interventions.
                        segseed=seed*1000+cutoff
                        pred=predict(g,actor,rh,data,segseed)
                        if cutoff in (42,82):
                            shift=(gt_history[0,-1,TRANSL]-rh[0,-1,TRANSL]).tolist()
                            angle=float(wrap(yaw(gt_history)[0,-1]-yaw(rh)[0,-1]))
                            variants={'original':rh,'root_reset':move(rh,data['offsets'][None,1],shift),
                                'heading_reset':move(rh,data['offsets'][None,1],angle=angle),
                                'both_reset':move(rh,data['offsets'][None,1],shift,angle),'gt_history':gt_history}
                            for mode,history in variants.items():
                                future=pred if mode=='original' else predict(g,actor,history,data,segseed)
                                probes.append(dict(**base,seed=seed,cutoff=cutoff,mode=mode,
                                    metrics=score(future,history,target,gt_history)))
                        generated.append(pred[0]);rh=pred[:,-2:]
                    full=torch.cat(generated);target=feat[1,2:122]
                    p=full[:,TRANSL];q=target[:,TRANSL]
                    vp=torch.diff(p,dim=0);vq=torch.diff(q,dim=0);b=torch.arange(7,119,8,device=g.device)
                    e=p[b]-q[b];c=-((vp[b]-vq[b])*e).sum(-1)/(e.norm(dim=-1)+1e-9)
                    heading=wrap(yaw(full)-yaw(target)).abs()
                    rollouts.append(dict(**base,seed=seed,root_cm=float((p-q).norm(dim=-1).mean()*100),
                        first40_cm=float((p[:40]-q[:40]).norm(dim=-1).mean()*100),
                        last40_cm=float((p[-40:]-q[-40:]).norm(dim=-1).mean()*100),
                        boundary_step_cm=float(vp[b].norm(dim=-1).mean()*100),
                        boundary_correction_cm=float(c.mean()*100),correction_fraction=float((c>0).float().mean()),
                        heading_deg=float(heading.mean()*180/math.pi)))
            for filename,rows in [('injections.json',injections),('gradients.json',gradrows),
                                  ('probes.json',probes),('rollouts.json',rollouts),('witnesses.json',witnesses)]:
                save(filename,rows)
            print(model,item['split'],item['episode'],'complete',flush=True)
        del g
        torch.cuda.empty_cache()


if __name__=='__main__':
    run()
