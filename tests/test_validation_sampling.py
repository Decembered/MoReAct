from collections import Counter
from types import SimpleNamespace

import pytest

from moreact.validation import action_class, audit_splits, stratified_indices


def dataset(split='val', classes=40):
    records = [{'episode': 'G%03dT000A%03dR%03d' % (1 if split == 'train' else 2, k, ep),
                'split': split} for k in range(classes) for ep in range(3)]
    return SimpleNamespace(records=records, manifest={'records': records},
                           windows=[(i, start) for i in range(len(records)) for start in range(20)])


def test_stratified_all_classes_unique_reproducible_and_rank_disjoint():
    d = dataset()
    ids = stratified_indices(d, 512, 9001)
    assert ids == stratified_indices(d, 512, 9001)
    assert ids != stratified_indices(d, 512, 9251)
    assert len(set(ids)) == 512
    counts = Counter(action_class(d.records[d.windows[i][0]]) for i in ids)
    assert len(counts) == 40 and set(counts.values()) == {12, 13}
    shards = [set(ids[i*128:(i+1)*128]) for i in range(4)]
    assert len(set.union(*shards)) == 512
    assert all(not a & b for i, a in enumerate(shards) for b in shards[i+1:])


def test_episode_balancing_and_small_budgets():
    d = dataset(classes=2)
    ids = stratified_indices(d, 6, 1)
    assert len({d.windows[i][0] for i in ids}) == 6
    with pytest.raises(ValueError, match='every action'):
        stratified_indices(d, 1, 1)
    with pytest.raises(ValueError, match='distinct'):
        stratified_indices(d, 121, 1)


def test_split_overlap_and_missing_classes_rejected():
    train, val = dataset('train'), dataset()
    assert len(audit_splits(train, val)['val_episodes_by_class']) == 40
    with pytest.raises(ValueError, match='coverage mismatch'):
        audit_splits(train, dataset(classes=39))
    val.manifest['records'].append({**train.records[0], 'split': 'val'})
    with pytest.raises(ValueError, match='overlap'):
        audit_splits(train, val)
