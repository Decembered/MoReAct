"""Reproducible class-balanced validation, with episode-level split checks."""
import re
from collections import Counter, defaultdict

import numpy as np


def action_class(record):
    match = re.fullmatch(r'G\d+T\d+A(\d{3})R\d+', record['episode'])
    if match is None:
        raise ValueError('Cannot infer Inter-X action class: ' + record['episode'])
    return 'A' + match[1]


def audit_splits(training, validation):
    manifest = validation.manifest
    train_ids = {r['episode'] for r in training.manifest['records'] if r['split'] == 'train'}
    val_ids = {r['episode'] for r in manifest['records'] if r['split'] == 'val'}
    overlap = train_ids & val_ids
    if overlap:
        raise ValueError('Train/val episode overlap: ' + ', '.join(sorted(overlap)))
    train_classes = Counter(action_class(r) for r in training.records)
    val_classes = Counter(action_class(r) for r in validation.records)
    if set(train_classes) != set(val_classes):
        raise ValueError('Train/val action coverage mismatch')
    return {'train_episodes_by_class': dict(sorted(train_classes.items())),
            'val_episodes_by_class': dict(sorted(val_classes.items())),
            'episode_overlap': [], 'separation_unit': 'episode (not subject)'}


def stratified_indices(dataset, count, seed):
    # Cache index groups, not sampled selections. Rank-independent local RNG.
    if not hasattr(dataset, '_validation_groups'):
        groups = defaultdict(lambda: defaultdict(list))
        for index, (episode, _) in enumerate(dataset.windows):
            groups[action_class(dataset.records[episode])][episode].append(index)
        dataset._validation_groups = dict(groups)
    groups = dataset._validation_groups
    classes = sorted(groups)
    if count < len(classes):
        raise ValueError('Validation sample count must cover every action class')
    rng = np.random.RandomState(seed % (2**32))
    quotas = {key: count // len(classes) for key in classes}
    for key in rng.permutation(classes)[:count % len(classes)]:
        quotas[key] += 1
    selected = []
    for key in classes:
        pools = {ep: list(rng.permutation(ids)) for ep, ids in groups[key].items()}
        if sum(map(len, pools.values())) < quotas[key]:
            raise ValueError('Not enough distinct validation windows for ' + key)
        remaining = quotas[key]
        while remaining:
            # Random episode cycles avoid favoring long episodes.
            for ep in rng.permutation(sorted(pools)):
                if pools[ep]:
                    selected.append(int(pools[ep].pop()))
                    remaining -= 1
                if remaining == 0:
                    break
    rng.shuffle(selected)
    return selected


def validation_indices(trainer):
    cfg = trainer.cfg['train']
    mode = cfg.get('val_sampling', 'sequential')
    count = cfg['batch_size'] * cfg['val_batches']
    if mode == 'sequential':
        return [i % len(trainer.validation) for i in range(count)]
    if mode != 'stratified_action':
        raise ValueError('Unknown validation sampling mode: ' + mode)
    seed = cfg['seed'] + 9001 + trainer.step
    indices = stratified_indices(trainer.validation, count, seed)
    if getattr(trainer, 'rank', 0) == 0:
        from .config import dump_json
        windows = []
        for index in indices:
            ep, start = trainer.validation.windows[index]
            record = trainer.validation.records[ep]
            windows.append({'index': index, 'episode': record['episode'],
                            'start': start, 'action': action_class(record)})
        dump_json(trainer.output / 'validation_samples' / ('step_%06d.json' % trainer.step),
                  {'mode': mode, 'seed': seed, 'step': trainer.step,
                   'class_counts': dict(sorted(Counter(w['action'] for w in windows).items())),
                   'unique_episodes': len({w['episode'] for w in windows}), 'windows': windows})
    return indices
