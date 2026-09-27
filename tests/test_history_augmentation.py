import pytest
import torch

from moreact.augmentation import translate_history, repair_target_boundary
from moreact.config import load_config, validate
from moreact.geometry import TRANSL, JOINTS, DTRANS, DJOINTS, make_features


def test_rigid_history_shift_and_target_boundary():
    torch.manual_seed(18)
    trans = torch.randn(8, 5, 3)
    rotations = torch.eye(3).expand(8, 5, 22, 3, 3)
    features = make_features(trans, rotations, trans[:, :, None].expand(-1, -1, 22, -1))
    history, target = features[:, :2], features[:, 2:]
    original = features.clone()
    shifted, selected = translate_history(history, probability=1., std_m=.02, max_m=.05)
    displacement = shifted[..., TRANSL] - history[..., TRANSL]
    torch.testing.assert_close(displacement[:, 0], displacement[:, 1])
    assert (displacement[..., 2] == 0).all()
    assert displacement[:, 0].norm(dim=-1).max() <= .050001
    assert displacement.abs().sum() > 0
    torch.testing.assert_close(
        (shifted[..., JOINTS] - history[..., JOINTS]).reshape(8, 2, 22, 3),
        displacement[:, :, None].expand(-1, -1, 22, -1), atol=3e-7, rtol=1e-5)
    torch.testing.assert_close(shifted[..., DTRANS], history[..., DTRANS])
    torch.testing.assert_close(shifted[..., DJOINTS], history[..., DJOINTS])
    repaired = repair_target_boundary(target, shifted, selected)
    torch.testing.assert_close(repaired[..., TRANSL], target[..., TRANSL])
    torch.testing.assert_close(repaired[..., JOINTS], target[..., JOINTS])
    torch.testing.assert_close(repaired[:, 0, DTRANS], target[:, 0, TRANSL] - shifted[:, -1, TRANSL])
    torch.testing.assert_close(repaired[:, 0, DJOINTS], target[:, 0, JOINTS] - shifted[:, -1, JOINTS])
    torch.testing.assert_close(repaired[:, 1:], target[:, 1:])
    torch.testing.assert_close(features, original)
    # Unselected samples preserve even preexisting generated-history mismatches.
    selected[::2] = False
    repaired = repair_target_boundary(target, shifted, selected)
    torch.testing.assert_close(repaired[::2], target[::2])


def test_disabled_augmentation_preserves_rng_and_inputs():
    history = torch.zeros(2, 2, 276)
    before = torch.get_rng_state()
    actual, selected = translate_history(history)
    assert actual is history and not selected.any()
    torch.testing.assert_close(before, torch.get_rng_state())


@pytest.mark.parametrize('section,key,value', [
    ('history_augmentation', 'probability', 1.1),
    ('history_augmentation', 'std_m', -1),
    ('history_augmentation', 'max_m', 0),
    ('diffusion_loss_options', 'feature_root_weight', float('nan')),
    ('diffusion_loss_options', 'distance_map_threshold_m', -1),
    ('diffusion_loss_options', 'recompute_target_deltas', 1),
    ('diffusion_loss_options', 'unknown', 1),
])
def test_invalid_augmentation_and_loss_options(section, key, value):
    cfg = load_config()
    cfg[section] = {key: value}
    with pytest.raises(ValueError):
        validate(cfg)
