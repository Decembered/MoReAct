from __future__ import annotations

import copy
from pathlib import Path

import torch
import pytest

from moreact.data import InterXDataset, condition_window, load_stats
from moreact.diffusion import LatentDiffusion
from moreact.generate import ReactionGenerator, rollout_arrays
from moreact.models import ReactionDenoiser, ReactionVAE
from moreact.train import Trainer, batch_at, load_checkpoint, rollout_probability, save_checkpoint
from moreact.geometry import transform_features


def test_curriculum(tiny_config):
    t = dict(stage1_steps=2, stage2_steps=4)
    assert [rollout_probability(i, t) for i in (0, 2, 4, 6, 9)] == [0., 0., .5, 1., 1.]


def test_checkpoint_statistics_mismatch_rejected():
    trainer = object.__new__(Trainer)
    trainer.stats = {"mean": torch.zeros(276), "std": torch.ones(276)}
    saved = {"stats": {k: v.clone() for k, v in trainer.stats.items()}}
    trainer.check_stats(saved)
    saved["stats"]["mean"][0] = 0.01
    with pytest.raises(ValueError, match="normalization differs"):
        trainer.check_stats(saved)


def test_new_objective_can_extend_primitives_but_ordinary_resume_cannot(tiny_config, tmp_path):
    vae = Trainer(tiny_config, 'vae', tmp_path/'vae')
    vae.checkpoint()
    first = Trainer(tiny_config, 'diffusion', tmp_path/'first',
                    vae_path=tmp_path/'vae/last.pt')
    first.checkpoint()
    longer = copy.deepcopy(tiny_config)
    longer['data']['primitives'] = 4
    with pytest.raises(ValueError, match='data.primitives'):
        Trainer(longer, 'diffusion', tmp_path/'first', resume=tmp_path/'first/last.pt')
    branched = Trainer(longer, 'diffusion', tmp_path/'branched',
                       resume=tmp_path/'first/last.pt', new_objective=True)
    assert branched.dataset.length == longer['data']['history'] + 4*longer['data']['future']
    assert branched.step == first.step and branched.best_val == float('inf')


@pytest.mark.parametrize('enhanced', [False, True])
def test_diffusion_geometry_backpropagates_through_frozen_vae(tiny_config, tmp_path, monkeypatch, enhanced):
    from moreact.losses import diffusion_motion_losses
    vae_trainer = Trainer(tiny_config, 'vae', tmp_path / 'geometry_vae')
    vae_trainer.checkpoint()
    tiny_config['diffusion_loss'] = dict(latent_mse=1., feature_rec=1.,
        smpl_joints_rec=10., joint_fk_consistency=10., joint_velocity=100.,
        bone_length=10., foot_contact=30., root_orientation=1., root_angular_velocity=10.,
        distance_map=1., joint_contact=10., root_position=30.)
    calls = []
    if enhanced:
        from moreact.config import load_config
        from moreact.augmentation import translate_history
        recipe = load_config('configs/diffusion_geometry.yaml')
        for key in ('diffusion_loss', 'diffusion_loss_options', 'history_augmentation'):
            tiny_config[key] = copy.deepcopy(recipe[key])
        tiny_config['history_augmentation']['probability'] = 1.
        def tracked_history(*args, **kwargs):
            result = translate_history(*args, **kwargs)
            calls.append(result[1])
            return result
        monkeypatch.setattr('moreact.train.translate_history', tracked_history)
    trainer = Trainer(tiny_config, 'diffusion', tmp_path / 'geometry_diff',
                      vae_path=tmp_path / 'geometry_vae/last.pt')
    batch = batch_at(trainer.dataset, [0, 1], 'cpu')
    a,r,origin,basis = condition_window(batch['actor'][:,:4], batch['reactor'][:,:4],
                                        batch['offsets'], trainer.stats)
    gt = transform_features(batch['reactor'][:,4:6],origin,basis,batch['offsets'][:,1])
    gt = (gt-trainer.stats['mean'])/trainer.stats['std']
    z = torch.randn(2,1,tiny_config['model']['latent_dim'], requires_grad=True)
    pred = trainer.vae.decode(z,a,r)
    loss, terms = diffusion_motion_losses(pred,gt,r,trainer.stats['mean'],trainer.stats['std'],
        trainer.body,batch['betas'][:,1],batch['genders'][:,1],tiny_config['diffusion_loss'],
        actor_future=transform_features(batch['actor'][:,4:6],origin,basis,batch['offsets'][:,0]),
        options=tiny_config.get('diffusion_loss_options'))
    loss.backward()
    assert z.grad is not None and torch.isfinite(z.grad).all() and z.grad.abs().sum()>0
    assert all(p.grad is None and not p.requires_grad for p in trainer.vae.parameters())
    before = copy.deepcopy(trainer.vae.state_dict())
    result = trainer.run(1)
    assert 'joint_velocity' in result and 'weighted_foot_contact' in result
    if enhanced:
        assert 'root_position' not in result and len(calls) == tiny_config['data']['primitives']
        assert all(mask.all() for mask in calls)
        original_batch = batch['reactor'].clone()
        # Exercise noisy training together with generated-history feedback.
        trainer.losses(batch, trainer.model, probability=1.)['loss'].backward()
        torch.testing.assert_close(batch['reactor'], original_batch)
        calls.clear()
        trainer.validate()
        with torch.no_grad():
            trainer.losses(batch, trainer.model, probability=1.)
        trainer.model.eval()
        trainer.losses(batch, trainer.model)
        trainer.model.train()
        assert not calls  # Validation and evaluation never inject history noise.
    else:
        assert result['root_position'] > 0 and result['weighted_root_position'] > 0
    assert abs(result['loss']-sum(v for k,v in result.items() if k.startswith('weighted_'))) < 1e-4
    for key,value in before.items():
        torch.testing.assert_close(value,trainer.vae.state_dict()[key],rtol=0,atol=0)
    changed = copy.deepcopy(tiny_config)
    changed['diffusion_loss']['foot_contact'] = 1.
    trainer.checkpoint()
    if enhanced:
        for section in ('diffusion_loss_options', 'history_augmentation'):
            mismatch = copy.deepcopy(tiny_config)
            mismatch.pop(section)
            with pytest.raises(ValueError, match=section + ' mismatch'):
                Trainer(mismatch, 'diffusion', tmp_path/'geometry_diff',
                        resume=tmp_path/'geometry_diff/last.pt')
    with pytest.raises(ValueError, match='diffusion_loss mismatch'):
        Trainer(changed,'diffusion',tmp_path/'geometry_diff',resume=tmp_path/'geometry_diff/last.pt')
    branch = Trainer(changed, 'diffusion', tmp_path/'new_objective',
                     resume=tmp_path/'geometry_diff/last.pt', new_objective=True)
    assert branch.step == trainer.step and branch.best_val == float('inf')
    for key, value in trainer.model.state_dict().items():
        torch.testing.assert_close(value, branch.model.state_dict()[key], rtol=0, atol=0)
    assert branch.optimizer.state_dict()['state'].keys() == trainer.optimizer.state_dict()['state'].keys()
    with pytest.raises(ValueError, match='new run'):
        Trainer(changed, 'diffusion', tmp_path/'geometry_diff',
                resume=tmp_path/'geometry_diff/last.pt', new_objective=True)


def test_vae_overfits_fixed_batch(tiny_config):
    torch.set_num_threads(1)
    torch.manual_seed(12)
    model = ReactionVAE(tiny_config)
    data = InterXDataset(tiny_config)
    batch = batch_at(data, [0, 1], "cpu")
    stats = load_stats(tiny_config["data"]["cache"], "cpu")
    a, r, origin, basis = condition_window(batch["actor"][:, :4], batch["reactor"][:, :4], batch["offsets"], stats)
    target = transform_features(batch["reactor"][:, 4:6], origin, basis, batch["offsets"][:, 1])
    target = (target - stats["mean"]) / stats["std"]
    opt = torch.optim.Adam(model.parameters(), lr=.003)
    losses = []
    for i in range(100):
        z, mu, logvar = model.encode(a, r, target)
        pred = model.decode(z, a, r)
        loss = (pred - target).square().mean() + 1e-4 * (mu.square() + logvar.exp() - logvar - 1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    assert sum(losses[-10:]) / 10 < losses[0] * .15


def test_resume_matches_uninterrupted_and_diffusion_freezes_vae(tiny_config, tmp_path):
    first = Trainer(tiny_config, "vae", tmp_path / "partial")
    first.run(2)
    resumed = Trainer(tiny_config, "vae", tmp_path / "partial", resume=tmp_path / "partial/last.pt")
    resumed.run(4)
    best = load_checkpoint(tmp_path / 'partial/best.pt')
    assert best['step'] in (2, 4)
    assert best['best_val'] == resumed.best_val
    whole = Trainer(tiny_config, "vae", tmp_path / "whole")
    whole.run(4)
    for key, value in whole.model.state_dict().items():
        torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    for key, value in whole.ema.state_dict().items():
        torch.testing.assert_close(value, resumed.ema.state_dict()[key], rtol=0, atol=0)
    diffusion = Trainer(tiny_config, "diffusion", tmp_path / "diff", vae_path=tmp_path / "whole/last.pt")
    before = copy.deepcopy(diffusion.vae.state_dict())
    diffusion.run(2)
    assert all(not p.requires_grad and p.grad is None for p in diffusion.vae.parameters())
    for key, value in before.items():
        torch.testing.assert_close(value, diffusion.vae.state_dict()[key], rtol=0, atol=0)
    resumed_diff = Trainer(tiny_config, "diffusion", tmp_path / "diff", resume=tmp_path / "diff/last.pt")
    resumed_diff.run(4)
    direct_diff = Trainer(tiny_config, "diffusion", tmp_path / "diff_whole", vae_path=tmp_path / "whole/last.pt")
    direct_diff.run(4)
    for key, value in direct_diff.model.state_dict().items():
        torch.testing.assert_close(value, resumed_diff.model.state_dict()[key], rtol=0, atol=0)


class IdentityBody:
    def model(self, gender):
        return None

    def repair(self, generated, previous, betas, genders):
        return generated


def test_generation_causality_and_streaming_feedback(tiny_config, tmp_path):
    cfg = tiny_config
    model, vae = ReactionDenoiser(cfg).eval(), ReactionVAE(cfg).eval()
    ckpt = tmp_path / "diffusion.pt"
    save_checkpoint(ckpt, {"kind": "diffusion", "config": cfg, "ema": model.state_dict(),
                          "vae": vae.state_dict(), "stats": load_stats(cfg["data"]["cache"], "cpu"),
                          "data_digest": "synthetic-only", "step": 0})
    data = InterXDataset(cfg)[0]

    def run(actor, reactor, frames):
        g = ReactionGenerator(ckpt, "cpu", seed=12)
        g.body = IdentityBody()
        return rollout_arrays(g, actor, reactor[:4], data["betas"], data["genders"],
                              data["offsets"], frames, data["text"][None])[0]

    original = run(data["actor"], data["reactor"], 2)
    actor, reactor = data["actor"].clone(), data["reactor"].clone()
    actor[4:] += 100
    reactor[4:] -= 100
    torch.testing.assert_close(original, run(actor, reactor, 2), rtol=0, atol=0)
    # Reactor truth after initialization has no route into any later segment.
    torch.testing.assert_close(run(data["actor"], data["reactor"], 4),
                               run(data["actor"], reactor, 4), rtol=0, atol=0)


def test_actor_condition_and_text_cfg_do_not_remove_history(tiny_config):
    torch.manual_seed(8)
    model = ReactionDenoiser(tiny_config).eval()
    a, r = torch.randn(2, 4, 276), torch.randn(2, 4, 276)
    z, t, text = torch.randn(2, 1, 8), torch.tensor([1, 2]), torch.randn(2, 512)
    pred = model(z, t, a, r, text, force_no_text=True)
    assert not torch.allclose(pred, model(z, t, torch.zeros_like(a), r, text, force_no_text=True))
    torch.testing.assert_close(pred, model(z, t, a, r, torch.zeros_like(text)))
    diffusion = LatentDiffusion(3)
    sample = diffusion.sample(model, a, r, text, 8, guidance=2.)
    assert sample.shape == (2, 1, 8) and torch.isfinite(sample).all()
