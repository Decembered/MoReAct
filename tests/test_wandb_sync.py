import io
import pytest
from scripts.sync_diffusion_wandb import metric_payload, complete_records


def test_all_loss_groups_and_weights_are_distinct():
    terms = ['latent_mse', 'feature_rec', 'smpl_joints_rec', 'joint_fk_consistency',
             'joint_velocity', 'bone_length', 'foot_contact', 'root_orientation',
             'root_position', 'root_angular_velocity', 'distance_map', 'joint_contact']
    record = {'step': 10501, 'learning_rate': .0001}
    for prefix in ['', 'val_', 'val_rollout_']:
        record[prefix+'loss'] = 3.
        for term in terms:
            record[prefix+term] = .1
            record[prefix+'weighted_'+term] = .2
    payload = metric_payload(record)
    for prefix in ['train', 'val', 'val_rollout']:
        assert payload[prefix+'/loss'] == 3.
        for term in terms:
            assert payload[prefix+'/raw/'+term] == .1
            assert payload[prefix+'/weighted/'+term] == .2
    assert len(payload) == len(record)
    with pytest.raises(ValueError):
        metric_payload({'step': 1, 'loss': float('nan')})


def test_partial_jsonl_line_is_retried():
    stream = io.StringIO('{"step":1}\n{"step":')
    assert list(complete_records(stream)) == [{'step': 1}]
    start = stream.tell()
    stream.seek(0, 2);stream.write('2}\n');stream.seek(start)
    assert list(complete_records(stream)) == [{'step': 2}]
