"""Regression checks for diffusion history targets and single-device training."""
import copy
from types import SimpleNamespace

import pytest
import torch

from moreact.geometry import DTRANS, DJOINTS, TRANSL, JOINTS, POSE, DROT, rotation_6d_to_matrix, matrix_to_6d
from moreact.train import Trainer, batch_at, load_checkpoint, learning_rate_at


def test_generated_history_targets_recompute_deltas_without_noise(tiny_config, tmp_path, monkeypatch):
    vae = Trainer(tiny_config, 'vae', tmp_path/'vae')
    vae.checkpoint()
    tiny_config['diffusion_loss_options'] = {'recompute_target_deltas': True}
    trainer = Trainer(tiny_config, 'diffusion', tmp_path/'diff', vae_path=tmp_path/'vae/last.pt')
    batch = batch_at(trainer.dataset, [0, 1], 'cpu')
    original = batch['reactor'].clone()
    observed = []
    original_encode = trainer.vae.encode
    def check_encode(a, r, target):
        history = r * trainer.stats['std'] + trainer.stats['mean']
        future = target * trainer.stats['std'] + trainer.stats['mean']
        for positions, deltas in ((TRANSL, DTRANS), (JOINTS, DJOINTS)):
            torch.testing.assert_close(future[:, 0, deltas],
                future[:, 0, positions] - history[:, -1, positions], atol=2e-5, rtol=1e-5)
        root = rotation_6d_to_matrix(future[:, 0, POSE.start:POSE.start+6])
        previous = rotation_6d_to_matrix(history[:, -1, POSE.start:POSE.start+6])
        torch.testing.assert_close(future[:, 0, DROT], matrix_to_6d(root @ previous.transpose(-1, -2)),
                                   atol=2e-5, rtol=1e-5)
        observed.append(target)
        return original_encode(a, r, target)
    monkeypatch.setattr(trainer.vae, 'encode', check_encode)
    # No augmentation config, eval mode, and no gradients: this covers rollout validation.
    with torch.no_grad():
        trainer.losses(batch, trainer.ema, probability=1.)
    assert len(observed) == tiny_config['data']['primitives']
    torch.testing.assert_close(original, batch['reactor'])


def test_single_device_validation_honors_rollout_and_preserves_rng(tiny_config, monkeypatch):
    trainer = object.__new__(Trainer)
    trainer.cfg = copy.deepcopy(tiny_config)
    trainer.cfg['train'].update(validate_rollout=True, val_batches=2)
    trainer.step, trainer.device = 0, torch.device('cpu')
    trainer.validation = list(range(10))
    trainer.ema = SimpleNamespace(training=False)
    probabilities = []
    def losses(batch, model, probability=0.):
        assert not torch.is_grad_enabled() and not model.training
        probabilities.append(probability)
        torch.rand(3)
        return {'loss': torch.tensor(1. + probability)}
    trainer.losses = losses
    monkeypatch.setattr('moreact.train.batch_at', lambda *args: {})
    before = torch.get_rng_state()
    values = trainer.validate()
    assert values == {'val_loss': 1., 'val_rollout_loss': 2.}
    assert probabilities.count(0.) == probabilities.count(1.) == 2
    torch.testing.assert_close(before, torch.get_rng_state())


def test_single_device_lr_origin_and_rollout_best_resume(tiny_config, tmp_path):
    vae = Trainer(tiny_config, 'vae', tmp_path/'vae')
    vae.checkpoint()
    tiny_config['train'].update(lr_schedule_start_step=1, validate_rollout=True)
    trainer = Trainer(tiny_config, 'diffusion', tmp_path/'diff', vae_path=tmp_path/'vae/last.pt')
    result = trainer.run(2)  # Last update at step 1 must use the base LR.
    assert trainer.optimizer.param_groups[0]['lr'] == pytest.approx(tiny_config['train']['learning_rate'])
    best = load_checkpoint(tmp_path/'diff/best_rollout.pt')
    assert best['best_rollout'] == result['val_rollout_loss']
    resumed = Trainer(tiny_config, 'diffusion', tmp_path/'diff', resume=tmp_path/'diff/last.pt')
    assert resumed.best_rollout == best['best_rollout']
    branch = Trainer(tiny_config, 'diffusion', tmp_path/'branch',
                     resume=tmp_path/'diff/last.pt', new_objective=True)
    assert branch.best_val == branch.best_rollout == float('inf')


def test_lr_schedule_limits_and_invalid_origins(tiny_config):
    from moreact.config import validate
    cfg = tiny_config['train']
    cfg['lr_schedule_start_step'] = 2
    base = cfg['learning_rate']
    assert [learning_rate_at(s, cfg) for s in (0, 2, 6, 10, 20)] == [base, base, base*.5, 0., 0.]
    for origin in (-1, 10, 1.5, True):
        cfg['lr_schedule_start_step'] = origin
        with pytest.raises(ValueError, match='lr_schedule_start_step'):
            validate(tiny_config)
