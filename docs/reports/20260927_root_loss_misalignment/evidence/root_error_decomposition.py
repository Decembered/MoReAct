"""Decompose rollout root error into loss-visible (segment-local, anchored) and
anchor/accumulation components. Mirrors reference_frame() from moreact/geometry.py:
basis x = hip axis (joints[2] - joints[1]) projected to the ground plane, z = up.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = '/data/autovla/projects/MoReAct/outputs/three_model_sample_20260927'
F, H, FRAMES = 8, 2, 120
BLOCKS = [(0, 40), (40, 80), (80, 120)]


def yaw_frame(joints, index):
    """2D heading from the hip axis at one frame; returns R (3,3) with x=yaw dir."""
    hips = joints[index, 2] - joints[index, 1]
    angle = np.arctan2(hips[1], hips[0])
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def metrics(p, q):
    """p, q: (T,22,3) world joints. Returns the diagnostic dictionary."""
    out = {}
    root = np.linalg.norm(p[:, 0] - q[:, 0], axis=-1)
    out['root_ade_cm'] = float(root.mean() * 100)
    out['root_fde_cm'] = float(root[-1] * 100)
    out['root_xy_ade_cm'] = float(np.linalg.norm(p[:, 0, :2] - q[:, 0, :2], axis=-1).mean() * 100)
    out['root_z_mae_cm'] = float(np.abs(p[:, 0, 2] - q[:, 0, 2]).mean() * 100)
    out['root_axis_mae_cm'] = float(np.abs(p[:, 0] - q[:, 0]).mean() * 100)
    out['root_xy_axis_mae_cm'] = float(np.abs(p[:, 0, :2] - q[:, 0, :2]).mean() * 100)
    for start, end in BLOCKS:
        out['root_ade_cm_block_%d_%d' % (start, end)] = float(root[start:end].mean() * 100)
    out['world_mpjpe_cm'] = float(np.linalg.norm(p - q, axis=-1).mean() * 100)
    out['root_aligned_mpjpe_cm'] = float(
        np.linalg.norm((p - p[:, :1]) - (q - q[:, :1]), axis=-1).mean() * 100)

    yaw_pred = np.array([np.arctan2(*(j[2] - j[1])[[1, 0]]) for j in p])
    yaw_gt = np.array([np.arctan2(*(j[2] - j[1])[[1, 0]]) for j in q])
    diff = np.arctan2(np.sin(yaw_pred - yaw_gt), np.cos(yaw_pred - yaw_gt))
    out['heading_abs_err_deg'] = float(np.abs(diff).mean() * 180 / np.pi)
    out['heading_signed_bias_deg'] = float(diff.mean() * 180 / np.pi)
    for start, end in BLOCKS:
        out['heading_signed_bias_deg_%d_%d' % (start, end)] = float(
            diff[start:end].mean() * 180 / np.pi)

    local_err, local_err_blocks = [], {b: [] for b in BLOCKS}
    damp_pred, damp_gt, along_err, angle_err, jumps, inside = [], [], [], [], [], []
    local_per_frame = {b: [] for b in BLOCKS}
    for m in range(FRAMES // F):
        t0 = 8 * m
        anchor = 8 * m - 1                       # last history frame before this segment
        if anchor < 0:
            anchor = 0                           # first segment: anchor is the initial GT frame
            Rp = yaw_frame(q, anchor)
            op = oq = q[anchor, 0].copy()
        else:
            Rp = yaw_frame(p, anchor)            # rollout anchors on generated history
            op, oq = p[anchor, 0].copy(), q[anchor, 0].copy()
        for t in range(t0, min(t0 + F, FRAMES)):
            rp = Rp.T @ (p[t, 0] - op)
            rq = Rp.T @ (q[t, 0] - oq)
            err = np.linalg.norm(rp - rq) * 100
            local_err.append(err)
            for b in BLOCKS:
                if b[0] <= t < b[1]:
                    local_err_blocks[b].append(err)
                    local_per_frame[b].append((t, err))
        # net displacement over the segment, in the anchor frame
        if anchor >= 0:
            dp = Rp.T @ (p[min(t0 + F - 1, FRAMES - 1), 0] - op)
            dq = Rp.T @ (q[min(t0 + F - 1, FRAMES - 1), 0] - oq)
            damp_pred.append(np.linalg.norm(dp) * 100)
            damp_gt.append(np.linalg.norm(dq) * 100)
            if np.linalg.norm(dq) > 1e-6:
                along_err.append(float(dp @ dq / np.linalg.norm(dq) * 100))
                angle_err.append(float(np.degrees(np.arctan2(dp[0] * dq[1] - dp[1] * dq[0],
                                                            dp[:2] @ dq[:2] + 1e-12))))
            if 0 < t0 < FRAMES:
                step = np.linalg.norm(p[t0, 0] - p[t0 - 1, 0]) * 100
                jumps.append(step)
                inside.append(np.linalg.norm(np.diff(p[max(0, t0 - F + 1):t0, 0], axis=0).mean(0)) * 100)
    out['loss_visible_local_root_err_cm'] = float(np.mean(local_err))
    for b, vals in local_err_blocks.items():
        out['loss_visible_local_root_err_cm_block_%d_%d' % b] = float(np.mean(vals))
    late = [v for b in [(80, 120)] for _, v in local_per_frame[b]]
    out['loss_visible_local_root_err_cm_last40'] = float(np.mean(late))
    out['segment_disp_pred_cm'] = float(np.mean(damp_pred))
    out['segment_disp_gt_cm'] = float(np.mean(damp_gt))
    out['segment_disp_ratio'] = float(np.mean(damp_pred) / max(np.mean(damp_gt), 1e-9))
    out['segment_along_err_signed_cm'] = float(np.mean(along_err))
    out['segment_direction_err_deg'] = float(np.mean(np.abs(angle_err)))
    out['boundary_step_cm'] = float(np.mean(jumps)) if jumps else float('nan')
    out['interior_step_cm'] = float(np.mean(inside)) if inside else float('nan')
    out['boundary_over_interior'] = float(np.mean(jumps) / max(np.mean(inside), 1e-9)) if jumps else float('nan')

    # transplanted trajectories: GT local motion on predicted anchor, and predicted
    # local motion on GT anchor. Isolates accumulation from segment prediction.
    pa, pb = p.copy(), p.copy()
    for m in range(FRAMES // F):
        t0 = 8 * m
        anchor = max(8 * m - 1, 0)
        Rp, Rq = yaw_frame(p, anchor), yaw_frame(q, anchor)
        op, oq = p[anchor, 0].copy(), q[anchor, 0].copy()
        for t in range(t0, min(t0 + F, FRAMES)):
            pa[t, :, :] = (Rp @ (q[t, :, :] - oq).T).T + op          # GT local + pred anchor
            pb[t, :, :] = (Rq @ (p[t, :, :] - op).T).T + oq          # pred local + GT anchor
    out['world_mpjpe_cm_gt_local_pred_anchor'] = float(np.linalg.norm(pa - q, axis=-1).mean() * 100)
    out['root_ade_cm_gt_local_pred_anchor'] = float(np.linalg.norm(pa[:, 0] - q[:, 0], axis=-1).mean() * 100)
    out['world_mpjpe_cm_pred_local_gt_anchor'] = float(np.linalg.norm(pb - q, axis=-1).mean() * 100)
    out['root_ade_cm_pred_local_gt_anchor'] = float(np.linalg.norm(pb[:, 0] - q[:, 0], axis=-1).mean() * 100)
    return out


def summarize(rows):
    keys = rows[0].keys()
    return {k: float(np.mean([r[k] for r in rows])) for k in keys}


def main():
    result = {}
    for model in ('gpu467', 'gpu0123'):
        rows = []
        for path in sorted(glob.glob(os.path.join(ROOT, model, '*.npz'))):
            z = np.load(path, allow_pickle=True)
            rows.append(metrics(z['prediction'].astype(np.float64), z['target'].astype(np.float64)))
        result[model] = {'episodes': len(rows), 'mean': summarize(rows)}
        print('==', model, 'episodes', len(rows))
        for k, v in result[model]['mean'].items():
            print('   %-45s %8.3f' % (k, v))
    out = sys.argv[1]
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print('written', out)


if __name__ == '__main__':
    main()
