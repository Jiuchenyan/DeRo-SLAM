import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局绘图样式（IEEE 期刊标准） =====================
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'

# ===================== 工具1：读取 KITTI 位姿 =====================
# KITTI 格式：每行12个数，3x4 变换矩阵行优先
def load_kitti_poses(file_path):
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            nums = list(map(float, line.strip().split()))
            if len(nums) != 12:
                continue
            # 提取平移分量 x, y, z
            x, y, z = nums[3], nums[7], nums[11]
            poses.append([x, y, z])
    return np.array(poses)

# ===================== 工具2：修正版 Sim(3) Umeyama 对齐 =====================
# 完全复现 evo 的 --align --correct_scale 逻辑
def umeyama_alignment(src, dst):
    """
    将源轨迹 src 通过 Sim(3) 变换对齐到目标轨迹 dst
    :param src: 待对齐轨迹 (N, 3)，即估计值
    :param dst: 参考轨迹 (N, 3)，即真值
    :return: aligned_src 对齐后的轨迹 (N,3), scale 尺度因子
    """
    assert src.shape == dst.shape, "轨迹长度必须一致"
    n = src.shape[0]

    # 1. 计算均值
    src_mean = np.mean(src, axis=0)
    dst_mean = np.mean(dst, axis=0)

    # 2. 去均值
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # ========== 之前的 bug 在这里：协方差矩阵方向写反了 ==========
    # 正确：H = (1/n) * dst^T @ src  （目标转置 × 源）
    H = (dst_centered.T @ src_centered) / n

    # 3. SVD 分解
    U, S, Vt = np.linalg.svd(H)

    # 4. 计算旋转矩阵，处理反射退化
    R = U @ Vt
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = U @ Vt

    # 5. 计算尺度因子
    var_src = np.mean(np.sum(src_centered ** 2, axis=1))
    scale = np.sum(S) / var_src

    # 6. 计算平移
    t = dst_mean - scale * (R @ src_mean)

    # 7. 应用 Sim(3) 变换
    aligned_src = scale * (src @ R.T) + t

    return aligned_src, scale

# ===================== 主流程 =====================
if __name__ == "__main__":
    # ---------- 1. 修改你的文件路径 ----------
    sequence = '08'
    gt_path = "/media/heu/data/Study/Library/data_base/KITTI/kitti_odometry/poses/"+sequence+".txt"
    # est_path = "/home/heu/ros/projs/DeRo-SLAM/src/ros/save/evo_log/kitti_mapping_"+sequence+".txt"
    est_path = "/home/heu/ros/projs/DeRo-SLAM/src/ros/save/debug_poses.txt"
    save_img_path = "kitti_"+sequence+"_ape_xz.png"

    # ---------- 2. 加载轨迹，长度对齐 ----------
    gt_xyz = load_kitti_poses(gt_path)
    est_xyz = load_kitti_poses(est_path)
    
    min_len = min(len(gt_xyz), len(est_xyz))
    gt_xyz = gt_xyz[:min_len]
    est_xyz = est_xyz[:min_len]
    print(f"轨迹帧数：真值 {len(gt_xyz)}, 估计 {len(est_xyz)}")

    # ---------- 3. Sim(3) 对齐 ----------
    est_aligned, scale = umeyama_alignment(est_xyz, gt_xyz)

    # ---------- 4. 计算逐帧 APE ----------
    ape_error = np.linalg.norm(est_aligned - gt_xyz, axis=1)

    # ---------- 5. 输出统计指标（和 evo 完全一致） ----------
    print("\n" + "="*45)
    print("APE w.r.t. translation part (m)")
    print("(with Sim(3) Umeyama alignment)")
    print("-"*45)
    print(f"   max    {np.max(ape_error):.6f}")
    print(f"   mean   {np.mean(ape_error):.6f}")
    print(f"   median {np.median(ape_error):.6f}")
    print(f"   min    {np.min(ape_error):.6f}")
    print(f"   rmse   {np.sqrt(np.mean(ape_error**2)):.6f}")
    print(f"   sse    {np.sum(ape_error**2):.6f}")
    print(f"   std    {np.std(ape_error):.6f}")
    print(f"   scale  {scale:.6f}")
    print("="*45 + "\n")

    # ---------- 6. 绘制 xz 平面误差着色轨迹 ----------
    x_gt, z_gt = gt_xyz[:, 0], gt_xyz[:, 2]
    x_est, z_est = est_aligned[:, 0], est_aligned[:, 2]

    fig, ax = plt.subplots(figsize=(3.4, 3.0))

    # 真值虚线
    ax.plot(x_gt, z_gt, color='black', linestyle='--', linewidth=1.2,
            alpha=0.6, label='reference')

    # 估计轨迹按误差着色
    sc = ax.scatter(x_est, z_est, c=ape_error, cmap='jet',
                    s=1.5, linewidths=0, alpha=0.9)

    # 精细颜色条
    cbar = plt.colorbar(sc, ax=ax, fraction=0.045, pad=0.04)
    # cbar.set_label('APE (m)')
    # cbar.ax.tick_params(labelsize=9)

    ax.set_xlabel('x (m)', labelpad=0)
    ax.set_ylabel('z (m)', labelpad=1)

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