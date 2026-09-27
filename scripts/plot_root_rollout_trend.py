"""Plot paired train/validation errors from fixed autoregressive rollouts."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
from matplotlib import pyplot as plt


METRICS = (
    ('root_ade_m', 'Root ADE', '根节点平均位置误差'),
    ('root_fde_m', 'Root FDE', '根节点末帧位置误差'),
    ('root_aligned_mpjpe_m', 'Root-aligned MPJPE', '根节点对齐关节误差'),
)
COLORS = {'train': '#d97706', 'val': '#2563eb'}


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def collect(root: Path, source_val: Path) -> tuple[dict, dict]:
    grouped = {}
    sources = {}
    for split in ('train', 'val'):
        for path in sorted((root / f'root_{split}').glob('step_*/per_episode.jsonl')):
            rows = read_rows(path)
            if not rows:
                continue
            steps = {row['step'] for row in rows}
            if len(steps) != 1:
                raise ValueError(f'Mixed checkpoint steps in {path}')
            key = split, steps.pop()
            grouped[key] = rows
            sources[key] = str(path.resolve())
    source_rows = read_rows(source_val)
    for step in {row['step'] for row in source_rows}:
        key = 'val', step
        if key not in grouped:
            grouped[key] = [row for row in source_rows if row['step'] == step]
            sources[key] = str(source_val.resolve())

    scores = defaultdict(dict)
    accepted_sources = defaultdict(dict)
    episode_sets = {}
    for (split, step), rows in sorted(grouped.items()):
        by_episode = defaultdict(dict)
        for row in rows:
            if row['split'] != split or row['seed'] in by_episode[row['episode']]:
                raise ValueError(f'Invalid duplicate or split in {sources[(split, step)]}')
            by_episode[row['episode']][row['seed']] = row['metrics']['feedback']
        if len(by_episode) != 40 or any(set(seeds) != {0, 1, 2} for seeds in by_episode.values()):
            # A watcher may currently be writing this checkpoint's results.
            continue
        episodes = set(by_episode)
        if split in episode_sets and episodes != episode_sets[split]:
            raise ValueError(f'{split} episode selection changed at step {step}')
        episode_sets[split] = episodes
        scores[split][step] = {
            metric: sum(sum(seeds[seed][metric] for seed in (0, 1, 2)) / 3
                        for seeds in by_episode.values()) / 40 * 100
            for metric, _, _ in METRICS
        }
        accepted_sources[split][step] = sources[(split, step)]
    return scores, accepted_sources


def plot(scores: dict, steps: list[int], prefix: Path) -> None:
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['WenQuanYi Zen Hei', 'DejaVu Sans'],
        'axes.unicode_minus': False,
        'font.size': 11,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'savefig.facecolor': 'white',
    })
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    grid = fig.add_gridspec(2, 2, height_ratios=[1.18, 1], hspace=.45, wspace=.25)
    axes = [fig.add_subplot(grid[0, :]), fig.add_subplot(grid[1, 0]),
            fig.add_subplot(grid[1, 1])]
    label_steps = set(steps if len(steps) <= 5 else (steps[0], 32500, steps[-1]))
    tick_steps = steps if len(steps) <= 6 else [
        step for step in steps if step % 1000 == 0 or step in (steps[0], 32500, steps[-1])]
    for ax, (metric, english, chinese) in zip(axes, METRICS):
        for split, label in (('train', '训练集'), ('val', '验证集')):
            values = [scores[split][step][metric] for step in steps]
            ax.plot(steps, values, '-o', color=COLORS[split], linewidth=2.5,
                    markersize=5 if len(steps) > 8 else 7, label=label, zorder=3)
            if ax is axes[0]:
                for step, value in zip(steps, values):
                    if step not in label_steps:
                        continue
                    offset = (0, 10 if split == 'train' else -17)
                    ax.annotate(f'{value:.2f}', (step, value), xytext=offset,
                                textcoords='offset points', ha='center',
                                color=COLORS[split], fontsize=10, fontweight='bold')
        ax.axvline(32500, color='#8b94a3', linewidth=1.3, linestyle='--', zorder=1)
        ax.grid(axis='y', color='#e7ebf0', linewidth=1)
        ax.set_axisbelow(True)
        axis_ticks = tick_steps if ax is axes[0] else [
            step for step in tick_steps if step % 1000 == 0 or step in (steps[0], steps[-1])]
        ax.set_xticks(axis_ticks)
        ax.set_xticklabels([f'{step:,}' for step in axis_ticks])
        ax.set_xlim(min(steps) - 280, max(steps) + 280)
        ax.set_title(f'{chinese}  ({english})', loc='left', fontsize=12,
                     fontweight='bold', pad=12)
        ax.set_ylabel('误差 / cm')
        ax.set_xlabel('Checkpoint step')
        values = [scores[split][step][metric] for split in ('train', 'val') for step in steps]
        span = max(values) - min(values)
        pad = max(span * (.18 if ax is axes[0] else .15), .25)
        ax.set_ylim(min(values) - pad, max(values) + pad)
    axes[0].legend(loc='center left', bbox_to_anchor=(.025, .49), frameon=False, ncol=2)
    axes[0].text(32500 + 40, axes[0].get_ylim()[1] - .15, '八卡续训起点',
                 color='#6b7280', va='top', fontsize=10)
    fig.suptitle('120 帧纯自回归生成：训练集与验证集误差随 step 的变化',
                 fontsize=18, fontweight='bold', y=.985)
    fig.text(.5, .035,
             '每组固定 40 条序列 × 3 个随机种子；27000 为几何配方源权重，27500–32500 为四卡 batch 512，之后为八卡 batch 1024。'
             '数值越低越好；训练集与验证集为不同序列。',
             ha='center', color='#5b6472', fontsize=9)
    fig.subplots_adjust(left=.09, right=.97, top=.90, bottom=.12)
    for suffix in ('.png', '.svg'):
        fig.savefig(prefix.with_suffix(suffix), dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--source-val', required=True, type=Path)
    parser.add_argument('--output-prefix', required=True, type=Path)
    args = parser.parse_args()
    scores, sources = collect(args.root, args.source_val)
    steps = sorted(set(scores['train']) & set(scores['val']))
    if len(steps) < 2:
        raise ValueError('At least two complete paired checkpoints are required')
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    plot(scores, steps, args.output_prefix)
    args.output_prefix.with_suffix('.json').write_text(json.dumps({
        'frames': 120, 'episodes_per_split': 40, 'seeds': [0, 1, 2],
        'units': 'cm', 'steps': steps,
        'scores': {split: {str(step): scores[split][step] for step in steps}
                   for split in ('train', 'val')},
        'sources': {split: {str(step): sources[split][step] for step in steps}
                    for split in ('train', 'val')},
    }, indent=2) + '\n')
    print(', '.join(map(str, steps)))
    print(args.output_prefix.with_suffix('.png'))


if __name__ == '__main__':
    main()
