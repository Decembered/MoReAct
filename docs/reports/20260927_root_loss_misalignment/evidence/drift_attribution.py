"""Attribute the world root drift: per-frame velocity errors (boundary vs interior),
heading (yaw) mismatch vs speed/local residuals.
"""
import glob
import json
import os
import sys

import numpy as np

ROOT = '/data/autovla/projects/MoReAct/outputs/three_model_sample_20260927'
F, FRAMES = 8, 120


def hip_yaw(joints):
    hips = joints[:, 2, :2] - joints[:, 1, :2]
    return np.arctan2(hips[:, 1], hips[:, 0])


def rot(v, angle):
    """Rotate the ground-plane components by angle; height is unchanged."""
    c, s = np.cos(angle), np.sin(angle)
    return np.stack((c * v[:, 0] - s * v[:, 1], s * v[:, 0] + c * v[:, 1], v[:, 2]), -1)


def analyse(p, q):
    out = {}
    vp, vq = np.diff(p[:, 0], axis=0), np.diff(q[:, 0], axis=0)      # (119,3) frame velocity
    err = vp - vq                                                     # velocity error vector
    boundary = np.array([t for t in range(1, FRAMES) if t % F == 0]) - 1
    interior = np.setdiff1d(np.arange(len(err)), boundary)
    E = err.sum(0)
    out['final_error_cm'] = float(np.linalg.norm(E) * 100)
    out['first_frame_error_cm'] = float(np.linalg.norm(p[0, 0] - q[0, 0]) * 100)
    out['boundary_frame_count'] = int(len(boundary))
    out['boundary_velocity_error_sum_cm'] = float(np.linalg.norm(err[boundary].sum(0)) * 100)
    out['interior_velocity_error_sum_cm'] = float(np.linalg.norm(err[interior].sum(0)) * 100)
    share = float(E @ err[boundary].sum(0) / max(E @ E, 1e-18))
    out['boundary_share_of_final_error'] = share
    out['interior_share_of_final_error'] = 1 - share
    out['mean_velocity_error_boundary_cm'] = float(np.linalg.norm(err[boundary], axis=-1).mean() * 100)
    out['mean_velocity_error_interior_cm'] = float(np.linalg.norm(err[interior], axis=-1).mean() * 100)
    out['mean_speed_pred_cm'] = float(np.linalg.norm(vp, axis=-1).mean() * 100)
    out['mean_speed_gt_cm'] = float(np.linalg.norm(vq, axis=-1).mean() * 100)
    out['mean_speed_pred_boundary_cm'] = float(np.linalg.norm(vp[boundary], axis=-1).mean() * 100)
    out['mean_speed_gt_at_boundary_cm'] = float(np.linalg.norm(vq[boundary], axis=-1).mean() * 100)
    unit = vq / np.maximum(np.linalg.norm(vq, axis=-1, keepdims=True), 1e-9)
    along = (err * unit).sum(-1)
    out['along_track_error_sum_cm'] = float(along.sum() * 100)
    out['along_track_error_boundary_cm'] = float(along[boundary].sum() * 100)
    out['along_track_error_interior_cm'] = float(along[interior].sum() * 100)
    perp = err - along[:, None] * unit
    out['perp_velocity_sum_cm'] = float(np.linalg.norm(perp.sum(0)) * 100)
    out['perp_velocity_magnitude_cm'] = float(np.linalg.norm(perp, axis=-1).mean() * 100)
    # vector-sum detour (path length excess) versus net displacement
    out['path_length_pred_cm'] = float(np.linalg.norm(vp, axis=-1).sum() * 100)
    out['path_length_gt_cm'] = float(np.linalg.norm(vq, axis=-1).sum() * 100)
    out['net_disp_pred_cm'] = float(np.linalg.norm(p[-1, 0] - p[0, 0]) * 100)
    out['net_disp_gt_cm'] = float(np.linalg.norm(q[-1, 0] - q[0, 0]) * 100)
    # heading attribution: rotate GT velocity by the observed yaw mismatch
    yaw_err = hip_yaw(p) - hip_yaw(q)
    yaw_err = np.arctan2(np.sin(yaw_err), np.cos(yaw_err))
    H = (rot(vq, yaw_err[1:]) - vq).sum(0)
    residual = E - H
    out['heading_induced_drift_cm'] = float(np.linalg.norm(H) * 100)
    out['residual_drift_cm'] = float(np.linalg.norm(residual) * 100)
    out['heading_share_of_final_error'] = float(E @ H / max(E @ E, 1e-18))
    out['yaw_error_abs_deg'] = float(np.abs(yaw_err).mean() * 180 / np.pi)
    out['yaw_error_abs_deg_last40'] = float(np.abs(yaw_err[80:]).mean() * 180 / np.pi)
    out['yaw_error_abs_deg_first40'] = float(np.abs(yaw_err[:40]).mean() * 180 / np.pi)
    return out


def main():
    result = {}
    for model in ('gpu467', 'gpu0123'):
        rows = [analyse(np.load(path, allow_pickle=True)['prediction'].astype(np.float64),
                        np.load(path, allow_pickle=True)['target'].astype(np.float64))
                for path in sorted(glob.glob(os.path.join(ROOT, model, '*.npz')))]
        result[model] = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
        print('==', model, len(rows), 'episodes')
        for k, v in result[model].items():
            print('   %-40s %9.3f' % (k, v))
    with open(sys.argv[1], 'w') as f:
        json.dump(result, f, indent=2)
    print('written', sys.argv[1])


if __name__ == '__main__':
    main()
