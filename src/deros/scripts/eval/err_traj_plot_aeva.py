import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局绘图样式（IEEE 期刊标准） =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ===================== 工具1：通用位姿读取（兼容 KITTI / KITTI-360） =====================
def load_poses(file_path):
    """
    自动识别位姿格式：
    - 12列：标准 KITTI 里程计格式（3x4 变换矩阵行优先）
    - 13列：KITTI-360 格式（首列为帧索引，后12列为位姿）
    返回:
        xyz: 轨迹坐标 (N, 3)
        frame_ids: 帧索引数组 (N,)，12列格式时返回 0~N-1
    """
    data = np.loadtxt(file_path)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    
    col_num = data.shape[1]
    xyz_list = []
    frame_ids = []

    if col_num == 13:
        # KITTI-360 真值：首列帧号 + 后12列位姿（世界系：xy水平面，z高程）
        for row in data:
            frame_id = int(row[0])
            pose_12 = row[1:]
            x, y, z = pose_12[3], pose_12[7], pose_12[11]
            frame_ids.append(frame_id)
            xyz_list.append([x, y, z])
    elif col_num == 12:
        # 常规 SLAM 输出位姿
        for i, row in enumerate(data):
            x, y, z = row[3], row[7], row[11]
            frame_ids.append(i)
            xyz_list.append([x, y, z])
    else:
        raise ValueError(f"不支持的位姿格式，列数：{col_num}，仅支持12列/13列")

    return np.array(xyz_list), np.array(frame_ids)

# ===================== 工具2：修正版 Sim(3) Umeyama 对齐 =====================
def umeyama_alignment(src, dst):
    """
    Sim(3) 对齐，完全复现 evo --align --correct_scale 效果
    :param src: 待对齐轨迹 (N, 3)
    :param dst: 参考轨迹 (N, 3)
    :return: aligned_src, scale
    """
    assert src.shape == dst.shape, "匹配后轨迹长度必须一致"
    n = src.shape[0]

    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    H = (dst_centered.T @ src_centered) / n
    U, S, Vt = np.linalg.svd(H)

    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    var_src = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.sum(S) / var_src

    t = dst_mean - scale * (R @ src_mean)
    aligned_src = scale * (src @ R.T) + t

    return aligned_src, scale

# ===================== 主流程 =====================
if __name__ == "__main__":
    # ---------- 1. 修改路径与序列 ----------
    sequence = 'Building61_NorthGate_Track'
    # KITTI-360 真值（13列，首列帧号，世界坐标系）

    gt_path = f"/media/heu/data/Study/Library/data_base/My_Data/sequences/{sequence}/pose/lidar_poses.txt"
    # 你的算法输出轨迹（12列，帧号从0开始）
    est_path = f"/home/heu/ros/projs/DeRo-SLAM/src/ros/save/evo_log/our_dataset_mapping_{sequence}.txt"
    save_img_path = f"aeva_{sequence}_ape_xy.png"

    # ---------- 2. 读取轨迹 ----------
    gt_xyz, gt_frame_ids = load_poses(gt_path)
    est_xyz, est_frame_ids = load_poses(est_path)
    est_total_len = len(est_xyz)
    print(f"真值帧数: {len(gt_xyz)}, 估计轨迹总帧数: {est_total_len}")

    # ---------- 3. 按真值帧号匹配估计轨迹对应帧 ----------
    valid_mask = (gt_frame_ids >= 0) & (gt_frame_ids < est_total_len)
    matched_gt = gt_xyz[valid_mask]
    matched_est = est_xyz[gt_frame_ids[valid_mask]]
    print(f"有效匹配帧数: {len(matched_gt)}")

    if len(matched_gt) < 4:
        raise ValueError("有效匹配点过少，无法完成对齐")

    # ---------- 4. Sim(3) Umeyama 对齐 ----------
    est_aligned, scale = umeyama_alignment(matched_est, matched_gt)

    # ---------- 5. 计算逐帧 APE 并输出统计 ----------
    ape_error = np.linalg.norm(est_aligned - matched_gt, axis=1)

    print("\n" + "="*45)
    print("APE w.r.t. translation part (m)")
    print("(with Sim(3) Umeyama alignment)")
    print("-"*45)
    print(f"   max    {np.max(ape_error):.6f}")
    print(f"   mean   {np.mean(ape_error):.6f}")
    print(f" median  {np.median(ape_error):.6f}")
    print(f"   min    {np.min(ape_error):.6f}")
    print(f"   rmse   {np.sqrt(np.mean(ape_error**2)):.6f}")
    print(f"   sse    {np.sum(ape_error**2):.6f}")
    print(f"   std    {np.std(ape_error):.6f}")
    print(f"   scale  {scale:.6f}")
    print("="*45 + "\n")

    # ---------- 6. 绘制 x-y 平面误差着色轨迹（世界系俯视图） ----------
    x_gt, y_gt = matched_gt[:, 0], matched_gt[:, 1]
    x_est, y_est = est_aligned[:, 0], est_aligned[:, 1]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    # 真值虚线
    ax.plot(x_gt, y_gt, color='black', linestyle='--', linewidth=1.2,
            alpha=0.6, label='reference')

    # 估计轨迹按误差着色
    sc = ax.scatter(x_est, y_est, c=ape_error, cmap='jet',
                    s=1.5, linewidths=0, alpha=0.9)

    # 精细颜色条
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
    # cbar.set_label('Err (m)')
    # cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel('x (m)', labelpad=0)
    ax.set_ylabel('y (m)', labelpad=1)

    # 缩小刻度数字与轴线的距离
    ax.tick_params(axis='both', pad=1)

    ax.set_aspect('auto')
    ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

    ax.legend(
        fontsize=8,
        framealpha=0.95,
        borderpad=0.2,
        labelspacing=0.1,
        handlelength=1.2
    )

    plt.tight_layout()
    plt.savefig(save_img_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存: {save_img_path}")
    plt.close()