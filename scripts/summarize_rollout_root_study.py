"""Episode-clustered summaries and paired comparisons for the root rollout study."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


METRICS = ('root_ade_m', 'root_fde_m', 'root_aligned_mpjpe_m',
           'pair_root_distance_error_m', 'boundary_velocity_jump_mps')


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def episode_values(rows, mode='feedback'):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['episode']].append(row['metrics'][mode])
    return {episode: {metric: float(np.mean([entry[metric] for entry in entries]))
                      for metric in METRICS}
            for episode, entries in grouped.items()}


def interval(values, random, draws):
    values = np.asarray(values, dtype=np.float64)
    samples = random.choice(values, (draws, len(values)), replace=True).mean(axis=1)
    return [float(x) for x in np.percentile(samples, (2.5, 97.5))]


def describe(values, random, draws):
    return {'mean': float(np.mean(values)), 'ci95': interval(values, random, draws),
            'episodes': len(values)}


def compare(baseline, candidate, random, draws):
    if set(baseline) != set(candidate):
        raise ValueError('Paired comparison requires identical episode sets')
    episodes = sorted(baseline)
    output = {'episodes': len(episodes), 'metrics': {}, 'per_action': {}}
    for metric in METRICS:
        base = np.asarray([baseline[e][metric] for e in episodes])
        other = np.asarray([candidate[e][metric] for e in episodes])
        difference = other-base
        indices = random.randint(len(episodes), size=(draws, len(episodes)))
        boot_base = base[indices].mean(axis=1)
        boot_other = other[indices].mean(axis=1)
        improvement = 1-boot_other/boot_base
        output['metrics'][metric] = {
            'baseline_mean': float(base.mean()), 'candidate_mean': float(other.mean()),
            'candidate_minus_baseline': describe(difference, random, draws),
            'relative_improvement': float(1-other.mean()/base.mean()),
            'relative_improvement_ci95': [float(x) for x in np.percentile(improvement, (2.5, 97.5))]}
    for episode in episodes:
        action = episode.split('A')[1][:3]
        output['per_action'][action] = {
            'episode': episode,
            'root_ade_baseline_m': baseline[episode]['root_ade_m'],
            'root_ade_candidate_m': candidate[episode]['root_ade_m'],
            'root_ade_difference_m': candidate[episode]['root_ade_m']-baseline[episode]['root_ade_m']}
    metric = output['metrics']
    output['recommendation_passes'] = bool(
        metric['root_ade_m']['relative_improvement'] >= .15 and
        metric['root_ade_m']['relative_improvement_ci95'][0] > 0 and
        metric['root_aligned_mpjpe_m']['candidate_minus_baseline']['mean'] <= .01 and
        metric['pair_root_distance_error_m']['candidate_minus_baseline']['mean'] <= .02 and
        metric['boundary_velocity_jump_mps']['candidate_minus_baseline']['mean'] <=
            .1*metric['boundary_velocity_jump_mps']['baseline_mean'])
    return output


def summarize_blocks(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for block in row['blocks']:
            for metric in ('root_ade_m', 'root_fde_m', 'along_track_delta_error_m',
                           'heading_error_deg', 'boundary_step_m', 'boundary_velocity_error_m'):
                groups[str(block['cutoff'])][metric].append(block[metric])
    return {cutoff: {metric: float(np.mean(values)) for metric, values in metrics.items()}
            for cutoff, metrics in sorted(groups.items(), key=lambda x: int(x[0]))}


def summarize_probes(rows, random, draws):
    grouped = defaultdict(lambda: defaultdict(list))
    for row in rows:
        by_cutoff = defaultdict(dict)
        for probe in row['probes']:
            by_cutoff[probe['cutoff']][probe['mode']] = probe
        for cutoff, modes in by_cutoff.items():
            for mode, probe in modes.items():
                grouped[(row['episode'], mode)]['root_ade_m'].append(probe['root_ade_m'])
                grouped[(row['episode'], mode)]['root_fde_m'].append(probe['root_fde_m'])
    modes = sorted({mode for _, mode in grouped})
    output = {}
    for mode in modes:
        if mode == 'original':
            continue
        episodes = sorted({episode for episode, name in grouped if name == mode})
        output[mode] = {}
        for metric in ('root_ade_m', 'root_fde_m'):
            deltas = [np.mean(grouped[(episode, mode)][metric])-
                      np.mean(grouped[(episode, 'original')][metric]) for episode in episodes]
            output[mode][metric+'_minus_original'] = describe(deltas, random, draws)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', action='append', required=True, help='LABEL=JSONL')
    parser.add_argument('--baseline')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=10000)
    args = parser.parse_args()
    random = np.random.RandomState(20260927)
    files = dict(item.split('=', 1) for item in args.file)
    rows = {label: read_rows(path) for label, path in files.items()}
    values = {label: episode_values(entries) for label, entries in rows.items()}
    report = {'inputs': files, 'summaries': {}, 'comparisons': {}}
    for label, entries in rows.items():
        report['summaries'][label] = {
            'rows': len(entries), 'episodes': len(values[label]),
            'feedback': {metric: describe([v[metric] for v in values[label].values()], random, args.draws)
                         for metric in METRICS},
            'teacher': {metric: describe([v[metric] for v in episode_values(entries, 'teacher').values()],
                                         random, args.draws) for metric in METRICS},
            'blocks': summarize_blocks(entries),
            'probes': summarize_probes(entries, random, args.draws)}
    if args.baseline:
        for label, entry in values.items():
            if label != args.baseline:
                report['comparisons'][label+'_vs_'+args.baseline] = compare(
                    values[args.baseline], entry, random, args.draws)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({'output': str(args.output),
                      'root_ade_m': {key: value['feedback']['root_ade_m']['mean']
                                     for key, value in report['summaries'].items()},
                      'comparisons': {key: value['metrics']['root_ade_m']
                                      for key, value in report['comparisons'].items()}}, indent=2))


if __name__ == '__main__':
    main()
