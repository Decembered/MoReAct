"""Finish A/B/C training and fixed validation/test evaluation after A and B launch.

Run on the study host. A and B are expected to have been started separately;
this script waits for their immutable 28500-step checkpoints, then trains C.
All long-running commands and failures are recorded in the study run directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/root_drift_short_20260927'
RESULTS = ROOT/'outputs/root_drift_short_20260927'
SOURCE = ROOT/'runs/diffusion_stage2_b512_stratified_20260926/step_027000.pt'
TEST_SELECTION = ROOT/'outputs/rollout_root_study_20260927/selection.json'
STEPS = (27500, 28000, 28500)


def checkpoint(branch, step):
    return RUN/branch/('step_%06d.pt' % step)


def record_status(stage, **extra):
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN/'pipeline_status.jsonl').open('a') as stream:
        stream.write(json.dumps({'time_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                 'stage': stage, **extra})+'\n')


def run_command(stage, command):
    log = RUN/(stage+'.log')
    record_status(stage, status='started', command=[str(x) for x in command], log=str(log))
    with log.open('a') as output:
        output.write('COMMAND: '+' '.join(str(x) for x in command)+'\n')
        output.flush()
        process = subprocess.run([str(x) for x in command], cwd=ROOT,
                                 stdout=output, stderr=subprocess.STDOUT,
                                 env={**os.environ, 'OMP_NUM_THREADS': '1'})
    record_status(stage, status='finished', returncode=process.returncode)
    if process.returncode:
        raise RuntimeError('%s failed; see %s' % (stage, log))


def wait_for_ab():
    deadline = time.monotonic()+8*3600
    while not all(checkpoint(branch, 28500).exists() for branch in ('A', 'B')):
        if time.monotonic() > deadline:
            raise TimeoutError('A/B did not reach step 28500 within eight hours')
        time.sleep(30)
    record_status('AB', status='checkpoints_ready')


def evaluate_one(branch, step, split, selection):
    folder = RESULTS/split/branch/str(step)
    path = folder/'per_episode.jsonl'
    run_command('eval_%s_%s_%s' % (split, branch, step),
                [sys.executable, '-u', 'scripts/rollout_root_study.py',
                 '--checkpoints', checkpoint(branch, step), '--output', folder,
                 '--split', split, '--selection', selection, '--device', 'cuda:7',
                 '--frames', 120, '--seeds', 0, 1, 2])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if len(rows) != 120 or len({r['episode'] for r in rows}) != 40:
        raise ValueError('%s has incomplete evaluation rows' % path)
    return path, rows


def select_val(rows):
    by_episode = {}
    for row in rows:
        by_episode.setdefault(row['episode'], []).append(row['metrics']['feedback']['root_ade_m'])
    return sum(sum(samples)/len(samples) for samples in by_episode.values())/len(by_episode)


def write_final_report(choices, outcome):
    target = ROOT/'docs/reports/20260927_rollout_root_reduction/final_results.md'
    lines = ['# A/B/C 短训结果', '',
             '固定 40 类 val 集各 1 条 episode、每条 3 个 seed 选择 checkpoint；',
             '每支只对选中 checkpoint 在固定 test 集评估一次。',
             '置信区间按 episode 聚类配对 bootstrap（10,000 次）。', '',
             '| 分支 | 选中 step | val root ADE | test root ADE | test FDE | 根对齐 MPJPE | 双人距离误差 | 边界跳变 |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for branch in ('A', 'B', 'C'):
        step = choices[branch]['selected_step']
        val = choices[branch]['val_root_ade_m'][step]
        fb = outcome['summaries'][branch]['feedback']
        lines.append('| %s | %d | %.4f m | %.4f m | %.4f m | %.4f m | %.4f m | %.4f m/s |' % (
            branch, step, val, fb['root_ade_m']['mean'], fb['root_fde_m']['mean'],
            fb['root_aligned_mpjpe_m']['mean'], fb['pair_root_distance_error_m']['mean'],
            fb['boundary_velocity_jump_mps']['mean']))
    lines += ['', '筛选结果（对 A）：', '']
    for branch in ('B', 'C'):
        item = outcome['comparisons'][branch+'_vs_A']
        root = item['metrics']['root_ade_m']
        ci = root['relative_improvement_ci95']
        lines.append('- %s：root ADE 相对改善 %.1f%%，95%% CI [%.1f%%, %.1f%%]；%s预设门槛。' % (
            branch, 100*root['relative_improvement'], 100*ci[0], 100*ci[1],
            '达到' if item['recommendation_passes'] else '未达到'))
    lines += ['', '逐类别 root ADE 差值（候选减 A，单位 cm）：', '',
              '| 类别 | B | C |', '| --- | ---: | ---: |']
    b = outcome['comparisons']['B_vs_A']['per_action']
    c = outcome['comparisons']['C_vs_A']['per_action']
    for action in sorted(b):
        lines.append('| %s | %+.2f | %+.2f |' % (
            action, 100*b[action]['root_ade_difference_m'],
            100*c[action]['root_ade_difference_m']))
    lines += ['', '完整配对结果见 `outputs/root_drift_short_20260927/test_summary.json`；',
              '全部逐 episode/seed 结果与 val 选择也保存在该目录。', '']
    target.write_text('\n'.join(lines))


def main():
    RESULTS.mkdir(parents=True, exist_ok=True)
    try:
        wait_for_ab()
        if not checkpoint('C', 28500).exists():
            run_command('train_C', [sys.executable, '-m', 'torch.distributed.run',
                '--standalone', '--nproc_per_node=4', '-m', 'moreact.train_distributed',
                '--config', RUN/'C.yaml', '--output', RUN/'C', '--resume', SOURCE,
                '--kind', 'diffusion', '--devices', '4,5,6,7', '--steps', '28500',
                '--new-objective'])
        if not checkpoint('C', 28500).exists():
            raise FileNotFoundError(checkpoint('C', 28500))
        val_selection = RESULTS/'val_selection.json'
        choices = {}
        for branch in ('A', 'B', 'C'):
            scores = {}
            for step in STEPS:
                _, rows = evaluate_one(branch, step, 'val', val_selection)
                scores[step] = select_val(rows)
            choices[branch] = {'val_root_ade_m': scores,
                               'selected_step': min(scores, key=scores.get)}
        (RESULTS/'val_choices.json').write_text(json.dumps(choices, indent=2)+'\n')
        record_status('val', status='selected', choices=choices)
        test_paths = {}
        for branch, item in choices.items():
            test_paths[branch], _ = evaluate_one(branch, item['selected_step'],
                                                 'test', TEST_SELECTION)
        summary = RESULTS/'test_summary.json'
        run_command('summarize_test', [sys.executable,
            'scripts/summarize_rollout_root_study.py',
            '--file', 'A='+str(test_paths['A']), '--file', 'B='+str(test_paths['B']),
            '--file', 'C='+str(test_paths['C']), '--baseline', 'A', '--output', summary])
        outcome = json.loads(summary.read_text())
        write_final_report(choices, outcome)
        record_status('complete', status='done', recommendations={
            name: value['recommendation_passes']
            for name, value in outcome['comparisons'].items()})
    except BaseException as exc:
        record_status('failure', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
