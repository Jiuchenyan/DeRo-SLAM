#!/usr/bin/env python3
"""
计算6自由度位姿误差详情：
- 平移 X, Y, Z 的 RMSE （单位：米）
- 欧拉角 (roll, pitch, yaw) 各分量的 RMSE （单位：度）
- 同时输出 KITTI 标准指标：平均平移误差、平均旋转误差、绝对轨迹误差、绝对旋转误差

用法:
    python3 detal_eval.py 04
    python3 detal_eval.py --gt_file <gt> --pred_file <pred> --calib_file <calib>
    python3 detal_eval.py 04 --pred_file custom_pred.txt
"""

import numpy as np
import argparse

from eval_traj_utils import (
    read_kitti_format_poses,
    read_kitti_format_calib,
    apply_kitti_format_calib,
    align_traj,
    absolute_error,
    relative_error,
)


def rot_to_euler_xyz(R):
    """将旋转矩阵转换为绕固定轴 X-Y-Z 的欧拉角 (roll, pitch, yaw)，单位：弧度"""
    if np.abs(R[2, 0]) < 1.0:
        pitch = -np.arcsin(R[2, 0])
        cos_p = np.cos(pitch)
        roll = np.arctan2(R[2, 1] / cos_p, R[2, 2] / cos_p)
        yaw = np.arctan2(R[1, 0] / cos_p, R[0, 0] / cos_p)
    else:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2.0
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2.0
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    return np.array([roll, pitch, yaw])


def six_dof_errors_euler(poses_gt, poses_result):
    n = len(poses_gt)
    trans_errors = np.zeros((n, 3))
    euler_errors = np.zeros((n, 3))

    for i in range(n):
        T_gt = poses_gt[i]
        T_pred = poses_result[i]
        delta = np.linalg.inv(T_gt) @ T_pred
        R_err = delta[:3, :3]
        t_err = delta[:3, 3]

        trans_errors[i] = t_err
        euler_errors[i] = rot_to_euler_xyz(R_err)

    euler_errors_deg = np.degrees(euler_errors)
    rmse = lambda x: np.sqrt(np.mean(x**2))

    results = {
        "tx_rmse (m)": rmse(trans_errors[:, 0]),
        "ty_rmse (m)": rmse(trans_errors[:, 1]),
        "tz_rmse (m)": rmse(trans_errors[:, 2]),
        "roll_rmse (deg)": rmse(euler_errors_deg[:, 0]),
        "pitch_rmse (deg)": rmse(euler_errors_deg[:, 1]),
        "yaw_rmse (deg)": rmse(euler_errors_deg[:, 2]),
    }
    return results


def main(gt_file, pred_file, calib_file):
    calib = read_kitti_format_calib(calib_file)
    gt_poses = read_kitti_format_poses(gt_file)
    pred_poses = read_kitti_format_poses(pred_file)
    pred_poses = apply_kitti_format_calib(pred_poses, calib["Tr"])

    gt_np = np.array(gt_poses)
    pred_np = np.array(pred_poses)

    rot_align, trans_align, _ = align_traj(pred_np, gt_np)
    T_align = np.eye(4)
    T_align[:3, :3] = rot_align
    T_align[:3, 3] = np.squeeze(trans_align)

    pred_aligned = np.array([T_align @ p for p in pred_np])

    rot_rmse_deg, tran_rmse_m, _ = absolute_error(gt_np, pred_np, align_on=True)
    drift_ate, drift_are = relative_error(gt_np, pred_np)

    errors = six_dof_errors_euler(gt_np, pred_aligned)

    print("\n============ Overall KITTI Metrics ============")
    print(f"Average Translation Error       (%): {drift_ate}")
    print(f"Average Rotational Error (deg/100m): {drift_are*100}")
    print(f"Absoulte Trajectory Error       (m): {tran_rmse_m}")
    print(f"Absoulte Rotation Error       (deg): {rot_rmse_deg}")

    print("\n============ 6-DoF Per-axis RMSE ============")
    for k, v in errors.items():
        print(f"{k}: {v:.6f}")
    print("=============================================\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='6-DoF detailed error evaluation')
    parser.add_argument('seq', nargs='?', default='05',help='KITTI sequence number (used if file paths are not given)')
    parser.add_argument('--gt_file', type=str, default=None,help='Path to ground truth poses file')
    parser.add_argument('--pred_file', type=str, default=None,help='Path to predicted odometry file')
    parser.add_argument('--calib_file', type=str, default=None,help='Path to calibration file')

    args = parser.parse_args()

    seq = "00"
    base_path = 'data/KITTI'

    gt_file = args.gt_file if args.gt_file else f'{base_path}/poses/{seq}.txt'
    pred_file = args.pred_file if args.pred_file else f'../save/evo_log/kitti_mapping_{seq}.txt'
    # pred_file = args.pred_file if args.pred_file else f'../save/evo_log/kitti_odometry_{seq}.txt'
    calib_file = args.calib_file if args.calib_file else f'{base_path}/calib/{seq}.txt'

    main(gt_file, pred_file, calib_file)