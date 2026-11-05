#!/usr/bin/env python3
"""
MulRan 位姿精度评估脚本
适配：
- 真值文件：帧号 + 12维KITTI位姿（每行13列）
- 预测文件：纯12维KITTI位姿（每行12列，按点云时间升序）

用法:
    python3 evaluate_mulran.py \
        --gt_file poses.txt \
        --pred_file your_pred_traj.txt \
        --seq riverside01 \
        --invert_pred
"""

import argparse
import numpy as np
from typing import List

# -------------------------------------------------------------------
# 核心评估函数（保持不变）
# -------------------------------------------------------------------

def align_traj(poses_np_1, poses_np_2):
    traj_1 = poses_np_1[:, :3, 3].squeeze().T
    traj_2 = poses_np_2[:, :3, 3].squeeze().T
    return align(traj_1, traj_2)


def align(model, data):
    """Closed-form alignment (Horn)"""
    model_zerocentered = model - model.mean(1, keepdims=True)
    data_zerocentered = data - data.mean(1, keepdims=True)

    W = np.zeros((3, 3))
    for column in range(model.shape[1]):
        W += np.outer(model_zerocentered[:, column], data_zerocentered[:, column])
    U, d, Vh = np.linalg.svd(W.transpose())
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vh) < 0:
        S[2, 2] = -1
    rot = U @ S @ Vh
    trans = data.mean(1, keepdims=True) - rot @ model.mean(1, keepdims=True)

    model_aligned = rot @ model + trans
    alignment_error = model_aligned - data
    trans_error = np.sqrt(np.sum(alignment_error ** 2, axis=0))
    return rot, trans, trans_error


def rotation_error(pose_error):
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    return np.arccos(max(min(d, 1.0), -1.0))


def translation_error(pose_error):
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    return np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)


def absolute_error(poses_gt: np.ndarray, poses_result: np.ndarray, align_on: bool = True):
    assert poses_gt.shape[0] == poses_result.shape[0]
    align_mat = np.eye(4)
    if align_on:
        align_rot, align_tran, _ = align_traj(poses_result, poses_gt)
        align_mat[:3, :3] = align_rot
        align_mat[:3, 3] = np.squeeze(align_tran)

    frame_count = poses_gt.shape[0]
    rot_errors = []
    tran_errors = []

    for i in range(frame_count):
        cur_aligned = align_mat @ poses_result[i]
        cur_gt = poses_gt[i]
        delta_rot = np.linalg.inv(cur_gt[:3, :3]) @ cur_aligned[:3, :3]
        delta_tran = cur_gt[:3, 3] - cur_aligned[:3, 3]

        rot_errors.append(rotation_error(delta_rot))
        tran_errors.append(np.linalg.norm(delta_tran))

    rot_errors = np.array(rot_errors)
    tran_errors = np.array(tran_errors)
    rot_rmse = np.sqrt(np.mean(rot_errors ** 2)) * 180.0 / np.pi
    tran_rmse = np.sqrt(np.mean(tran_errors ** 2))

    return rot_rmse, tran_rmse, align_mat


def trajectory_distances(poses_np):
    dist = [0]
    for i in range(poses_np.shape[0] - 1):
        rela_dist = np.linalg.norm(poses_np[i + 1, :3, 3] - poses_np[i, :3, 3])
        dist.append(dist[i] + rela_dist)
    return dist


def last_frame_from_segment_length(dist, first_frame, length):
    for i in range(first_frame, len(dist)):
        if dist[i] > (dist[first_frame] + length):
            return i
    return -1


def relative_error(poses_gt, poses_result):
    assert poses_gt.shape[0] == poses_result.shape[0]
    dist = trajectory_distances(poses_gt)
    step_size = 10
    lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    errors = []
    for first_frame in range(0, poses_gt.shape[0], step_size):
        for len_ in lengths:
            last_frame = last_frame_from_segment_length(dist, first_frame, len_)
            if last_frame == -1:
                continue

            pose_delta_gt = np.linalg.inv(poses_gt[first_frame]) @ poses_gt[last_frame]
            pose_delta_result = np.linalg.inv(poses_result[first_frame]) @ poses_result[last_frame]
            pose_error = np.linalg.inv(pose_delta_result) @ pose_delta_gt

            r_err = rotation_error(pose_error)
            t_err = translation_error(pose_error)
            num_frames = last_frame - first_frame + 1.0
            speed = len_ / (0.1 * num_frames)

            errors.append([first_frame, r_err / len_, t_err / len_, len_, speed])

    if len(errors) == 0:
        return 0.0, 0.0

    avg_r = np.mean([e[1] for e in errors])
    avg_t = np.mean([e[2] for e in errors])
    drift_ate = avg_t * 100.0
    drift_are = avg_r / np.pi * 180.0
    return drift_ate, drift_are


# -------------------------------------------------------------------
# 6-DoF 逐轴误差
# -------------------------------------------------------------------
def rot_to_euler_xyz(R):
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
        delta = np.linalg.inv(poses_gt[i]) @ poses_result[i]
        R_err = delta[:3, :3]
        t_err = delta[:3, 3]
        trans_errors[i] = t_err
        euler_errors[i] = rot_to_euler_xyz(R_err)

    euler_errors_deg = np.degrees(euler_errors)
    rmse = lambda x: np.sqrt(np.mean(x ** 2))

    return {
        "tx_rmse (m)": rmse(trans_errors[:, 0]),
        "ty_rmse (m)": rmse(trans_errors[:, 1]),
        "tz_rmse (m)": rmse(trans_errors[:, 2]),
        "roll_rmse (deg)": rmse(euler_errors_deg[:, 0]),
        "pitch_rmse (deg)": rmse(euler_errors_deg[:, 1]),
        "yaw_rmse (deg)": rmse(euler_errors_deg[:, 2]),
    }


# -------------------------------------------------------------------
# 数据读取函数
# -------------------------------------------------------------------
def read_kitti_format_poses(filename: str, invert=False) -> List[np.ndarray]:
    """读取纯12列KITTI格式位姿文件
    invert=True时返回位姿的逆矩阵
    """
    poses = []
    with open(filename) as f:
        for line in f:
            vals = [float(v) for v in line.strip().split()]
            pose = np.eye(4)
            pose[:3, :] = np.array(vals).reshape(3, 4)
            if invert:
                pose = np.linalg.inv(pose)
            poses.append(pose)
    return poses


def read_gt_with_frameid(pose_file: str):
    """
    读取带帧号的真值文件：第1列为雷达帧号（1-based），后续12列为3x4位姿矩阵
    """
    gt_poses = []
    frame_indices = []
    with open(pose_file) as f:
        for line in f:
            vals = line.strip().split()
            if len(vals) != 13:
                continue
            frame_idx = int(vals[0])
            mat = np.array(vals[1:], dtype=float).reshape(3, 4)
            T = np.eye(4)
            T[:3, :] = mat
            gt_poses.append(T)
            frame_indices.append(frame_idx)
    return np.array(gt_poses), frame_indices


# -------------------------------------------------------------------
# 主程序
# -------------------------------------------------------------------
# sequences = "kaist02"
# sequences = "dcc01"
sequences = "riverside02"
# sequences = "sejong01"
def main():
    parser = argparse.ArgumentParser(description='Evaluate SLAM on MulRan Dataset')
    parser.add_argument('--gt_file', type=str, default='/media/heu/data/Study/Library/data_base/mulran/'+sequences+'/poses/poses.txt', help='真值文件：帧号+12维位姿')
    # parser.add_argument('--pred_file', type=str, default='/home/heu/ros/projs/DeRo-SLAM/src/ros/save/evo_log/mulran_mapping_'+sequences+'.txt', help='预测文件：纯12维KITTI格式')
    parser.add_argument('--pred_file', type=str, default='/home/heu/ros/projs/SG-SLAM/src/ros/save/mulran_slam_poses_'+sequences+'.txt', help='预测文件：纯12维KITTI格式')
    # parser.add_argument('--pred_file', type=str, default='/home/heu/ros/projs/MULLS/result/mulran/pose_l_lo_mulls_mulran_'+sequences+'_bin.txt', help='预测文件：纯12维KITTI格式')
    # parser.add_argument('--pred_file',     type=str, default='/home/heu/ros/projs/saloam/result/mulran/mulran_'+sequences+'.txt',  help='file path of the ground truth odometry')
    # parser.add_argument('--pred_file', type=str, default='/home/heu/ros/projs/SAGE/results/mulran/mulran_optimized_'+sequences+'.txt',  help='file path of the ground truth odometry')
    # parser.add_argument('--pred_file', type=str, default='/home/heu/ros/projs/SAGE/results/mulran/mulran_no_foreground_optimized_'+sequences+'.txt',  help='file path of the ground truth odometry')

    parser.add_argument('--seq', type=str, default=sequences, help='序列名称')
    parser.add_argument('--invert_pred', action='store_true', help='对预测位姿取逆，`修正变换方向')
    args = parser.parse_args()

    # 读取真值与对应帧号
    print(f"Reading ground truth from {args.gt_file} ...")
    gt_poses, frame_indices = read_gt_with_frameid(args.gt_file)
    print(f"Number of GT poses: {len(gt_poses)}")
    print(f"GT frame range: {frame_indices[0]} ~ {frame_indices[-1]}")

    # 读取预测位姿
    print(f"\nReading predicted poses from {args.pred_file} ...")
    pred_all = read_kitti_format_poses(args.pred_file, invert=args.invert_pred)
    print(f"Total predicted frames: {len(pred_all)}")
    if args.invert_pred:
        print("已启用预测位姿取逆")

    # 按帧号匹配真值与预测
    valid_gt = []
    valid_pred = []
    skipped = 0
    for i, frame_idx in enumerate(frame_indices):
        pred_idx = frame_idx - 1
        if 0 <= pred_idx < len(pred_all):
            valid_gt.append(gt_poses[i])
            valid_pred.append(pred_all[pred_idx])
        else:
            skipped += 1

    if skipped > 0:
        print(f"\nWarning: {skipped} GT frames skipped (out of prediction range)")

    if len(valid_gt) == 0:
        raise RuntimeError("No overlapping frames between ground truth and prediction!")

    gt_np = np.array(valid_gt)
    pred_np = np.array(valid_pred)
    print(f"Valid evaluation frames: {len(valid_gt)}")

    # 误差计算
    rot_rmse_deg, tran_rmse_m, align_mat = absolute_error(gt_np, pred_np, align_on=True)
    drift_ate, drift_are = relative_error(gt_np, pred_np)
    pred_aligned = np.array([align_mat @ p for p in pred_np])
    axis_errors = six_dof_errors_euler(gt_np, pred_aligned)

    # 结果输出
    print("\n" + "="*50)
    print(f"Sequence:                    {args.seq}")
    print(f"Valid frames:               {len(valid_gt)}")
    print(f"ATE (RMSE) [m]:             {tran_rmse_m:.4f}")
    print(f"ARE (RMSE) [deg]:           {rot_rmse_deg:.4f}")
    print(f"KITTI RPE t_err [%]:        {drift_ate:.4f}")
    print(f"KITTI RPE r_err [deg/100m]: {drift_are:.4f}")
    print("\n--- Per-axis RMSE (after alignment) ---")
    for k, v in axis_errors.items():
        print(f"  {k}: {v:.4f}")
    print("="*50 + "\n")


if __name__ == '__main__':
    main()