import argparse
import hashlib
import json
import os
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path('/data/autovla/projects')
PROJECT = ROOT / 'MoReAct'
OUT = PROJECT / 'outputs/three_model_sample_20260927'
CACHE = PROJECT / 'data/interx_h2_f8'
PREPARED = ROOT / 'ttr_remogen_bridge/data/interx_smplx_full_10fps'
MODELS = {
    'gpu467': PROJECT / 'runs/diffusion_scratch_b768_root30_20260926/step_008500.pt',
    'gpu0123': PROJECT / 'runs/diffusion_stage2_b512_stratified_20260926/step_021000.pt',
}
sys.path.insert(0, str(PROJECT))


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def select():
    from moreact.geometry import POSE, rotation_6d_to_matrix
    manifest = json.loads((CACHE / 'manifest.json').read_text())
    rng = random.Random(20260927)
    groups = {}
    for r in manifest['records']:
        rid = r['episode'] + '_a' + r['actor_id'] + '_r' + r['reactor_id']
        if r['split'] in ('train', 'test') and r['frames'] >= 123 and (PREPARED / (rid+'.npz')).exists():
            action = r['episode'].split('A')[1][:3]
            groups.setdefault(r['split'], {}).setdefault(action, []).append((r, rid))
    actions = sorted(set(groups['train']) & set(groups['test']))
    rng.shuffle(actions)
    actions = actions[:20]
    selected = []
    for split in ('train', 'test'):
        for action in actions:
            r, rid = rng.choice(groups[split][action])
            with np.load(CACHE / r['file']) as data, np.load(PREPARED / (rid+'.npz')) as ref:
                assert len(ref['timestamps']) >= 123
                rotations = rotation_6d_to_matrix(torch.from_numpy(data['features'][0, :123, POSE].copy()).reshape(-1,22,6)).numpy()
                error = float(np.max(np.abs(rotations[:,1:] - ref['actor_body_pose'][:123])))
                assert error < 1e-4, (rid, error)
                selected.append(dict(record_id=rid, episode=r['episode'], actor_person=r['actor_id'],
                    reactor_person=r['reactor_id'], target_text=str(data['captions'][0]), split=split,
                    action=action, file=r['file'], source_rotation_max_error=error))
    dump(OUT / 'selection.json', selected)
    dump(OUT / 'protocol.json', dict(models={k:str(v) for k,v in MODELS.items()}, seed=0,
        selection_seed=20260927, frames=120, history=2, actions=actions,
        eligible={s:sum(map(len,g.values())) for s,g in groups.items()}, data_digest=manifest['digest']))
    print('Selected', len(selected), 'episodes across', len(actions), 'actions', flush=True)


def ours():
    from moreact.generate import ReactionGenerator, rollout_arrays
    from moreact.geometry import JOINTS
    selection = json.loads((OUT / 'selection.json').read_text())
    for name, checkpoint in MODELS.items():
        gen = ReactionGenerator(checkpoint, 'cuda:0', seed=0, guidance=1.)
        assert gen.data_digest == json.loads((CACHE/'manifest.json').read_text())['digest']
        for item in selection:
            dest = OUT / name / (item['record_id'] + '.npz')
            if dest.exists():
                continue
            with np.load(CACHE/item['file']) as z:
                data = {k:z[k] for k in z.files}
            gen.rng.manual_seed(0)
            pred, latency = rollout_arrays(gen, torch.from_numpy(data['features'][0,:122].copy()),
                torch.from_numpy(data['features'][1,:2].copy()), torch.from_numpy(data['betas']),
                torch.from_numpy(data['genders']), torch.from_numpy(data['offsets']), 120,
                torch.from_numpy(data['text_embeddings'][:1].copy()))
            dest.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest, prediction=pred.numpy()[:,JOINTS].reshape(-1,22,3),
                target=data['features'][1,2:122,JOINTS].reshape(-1,22,3),
                actor=data['features'][0,2:122,JOINTS].reshape(-1,22,3), latency=latency)
            print(name, item['split'], item['episode'], 'complete', flush=True)
        del gen
        torch.cuda.empty_cache()


def remogen():
    official_root = ROOT / 'remogen_official_release'
    sys.path.insert(0, str(official_root))
    os.chdir(official_root)
    from scripts.export_reaction100_official_mixed50 import prepare_dataset, run_one, official, Mixed50MLDArgs
    selection = json.loads((OUT/'selection.json').read_text())
    input_root = OUT/'official_input'
    prepare_dataset(selection, PREPARED, input_root)
    with (input_root/'test.pkl').open('rb') as f:
        records = pickle.load(f)
    for record in records:
        for motion in [record['motion'], *record['other_motion']]:
            for key in ('poses', 'trans', 'joints'):
                motion[key] = motion[key][:123]
    with (input_root/'test.pkl').open('wb') as f:
        pickle.dump(records, f)
    official.MLDArgs = Mixed50MLDArgs
    torch.set_num_threads(2)
    run_one('official_text', official_root/'checkpoints/interx_hhi_adapter/checkpoint.pt',
            input_root, OUT, selection, torch.device('cuda:0'))


def metrics(p, q, a):
    root = np.linalg.norm(p[:,0]-q[:,0], axis=-1)
    return dict(mpjpe_cm=float(np.linalg.norm(p-q,axis=-1).mean()*100),
        root_ade_cm=float(root.mean()*100), root_fde_cm=float(root[-1]*100),
        root_aligned_mpjpe_cm=float(np.linalg.norm((p-p[:,:1])-(q-q[:,:1]),axis=-1).mean()*100),
        pair_distance_error_cm=float(np.abs(np.linalg.norm(p[:,0]-a[:,0],axis=-1)-np.linalg.norm(q[:,0]-a[:,0],axis=-1)).mean()*100))


def summarize():
    selection = json.loads((OUT/'selection.json').read_text())
    rows = []
    residuals = []
    for item in selection:
        for name in (*MODELS, 'remogen'):
            if name == 'remogen':
                with (OUT/'official/text/seed_0'/ (item['record_id']+'.pkl')).open('rb') as f:
                    data = pickle.load(f)
                assert data['record_id'] == item['record_id']
                assert data['native_text'] == item['target_text'], (data['native_text'], item['target_text'])
                p,q,a = [data[k]['joints'][2:122] for k in ('reactor','gt_reactor','actor')]
                with np.load(OUT/'gpu467'/(item['record_id']+'.npz')) as reference:
                    X = np.concatenate([a.reshape(-1,3), q.reshape(-1,3)])
                    Y = np.concatenate([reference['actor'].reshape(-1,3), reference['target'].reshape(-1,3)])
                u, _, vt = np.linalg.svd((X-X.mean(0)).T @ (Y-Y.mean(0)))
                d = np.eye(3); d[-1,-1] = np.linalg.det(u @ vt)
                rotation = u @ d @ vt
                residuals.append(dict(episode=item['episode'], split=item['split'],
                    gt_rigid_alignment_residual_cm=float(np.linalg.norm((X-X.mean(0)) @ rotation-(Y-Y.mean(0)),axis=-1).mean()*100)))
            else:
                with np.load(OUT/name/(item['record_id']+'.npz')) as data:
                    p,q,a = [data[k] for k in ('prediction','target','actor')]
            assert p.shape == q.shape == a.shape == (120,22,3)
            assert all(np.isfinite(x).all() for x in (p,q,a))
            rows.append(dict(model=name, **item, metrics=metrics(p,q,a),
                first40=metrics(p[:40],q[:40],a[:40]), last40=metrics(p[-40:],q[-40:],a[-40:])))
    summary = {}
    for split in ('train','test'):
        summary[split] = {}
        for name in (*MODELS, 'remogen'):
            records = [r for r in rows if r['split']==split and r['model']==name]
            summary[split][name] = {k:float(np.mean([r['metrics'][k] for r in records])) for k in records[0]['metrics']}
    dump(OUT/'per_episode.json', rows)
    dump(OUT/'summary.json', summary)
    dump(OUT/'geometry_residuals.json', residuals)
    paired = {}
    rng = np.random.default_rng(20260927)
    for split in ('train','test'):
        paired[split] = {}
        left = [r for r in rows if r['split']==split and r['model']=='gpu467']
        for other in ('gpu0123','remogen'):
            right = [r for r in rows if r['split']==split and r['model']==other]
            assert [r['episode'] for r in left] == [r['episode'] for r in right]
            paired[split][other] = {}
            for key in left[0]['metrics']:
                delta = np.array([a['metrics'][key]-b['metrics'][key] for a,b in zip(left,right)])
                means = delta[rng.integers(0,len(delta),(10000,len(delta)))].mean(1)
                paired[split][other][key] = dict(gpu467_minus_other_mean=float(delta.mean()),
                    bootstrap95=np.percentile(means,[2.5,97.5]).tolist(), gpu467_wins=int((delta<0).sum()), n=len(delta))
    dump(OUT/'paired_comparison.json', paired)
    fingerprints = {name:dict(path=str(path), sha256=sha256(path))
                    for name,path in {**MODELS,
                        'remogen': ROOT/'remogen_official_release/checkpoints/interx_hhi_adapter/checkpoint.pt',
                        'remogen_vae': ROOT/'remogen_official_release/checkpoints/mvae_hml3d/checkpoint.pt'}.items()}
    dump(OUT/'checkpoints.json', fingerprints)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,3,figsize=(13,4),constrained_layout=True)
    for ax, key, title in zip(axes, ['mpjpe_cm','root_aligned_mpjpe_cm','pair_distance_error_cm'], ['World joint error','Root-aligned joint error','Pair distance error']):
        for i,(name,col) in enumerate(zip((*MODELS,'remogen'),['#168b81','#c15378','#5573bf'])):
            ax.bar(np.arange(2)+(i-1)*.25,[summary[s][name][key] for s in ('train','test')],.24,label=name,color=col)
        ax.set_xticks([0,1]); ax.set_xticklabels(['Train (20)','Test (20)'])
        ax.set_title(title); ax.set_ylabel('cm (lower is better)'); ax.grid(axis='y',alpha=.2)
    axes[0].legend()
    fig.suptitle('Paired sampled rollout | 120 future frames | seed 0 | native body geometry')
    fig.savefig(OUT/'comparison.png',dpi=160)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['select','ours','remogen','summarize'])
    globals()[parser.parse_args().stage]()
