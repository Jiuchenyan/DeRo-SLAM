#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Cylinder3D 在线语义推理节点
=====================================================================
移植自：semantic_liosam/src/kitti_odometry/scripts/cylinder3d_ros_node_dilute_step.py
改动点：
  1. 输入/输出都是普通 sensor_msgs/PointCloud2，话题由 rosparam 指定；
  2. 输出字段：x, y, z, intensity, ring, time, label(UINT16 原始 SemanticKITTI 标签), prob(UINT8)
     （原脚本把 label 写成可视化灰度颜色，DeRo 的 label_map 无法反解）；
  3. **工程根目录 / 权重路径 / 感知半径开关全部由 rosparam 传入**，不在代码里写死：
       ~cylinder3d_root    Cylinder3D 工程根目录（默认 /home/heu/ros/projs/semantic_liosam/src/cylinder3d）
       ~model_path         权重路径（相对 root 或绝对路径；空 = 用 config 里的 train_params.model_use_path）
       ~keep_out_of_range  超出模型感知半径的点是否发布（label=0，保留几何）
       ~debug              打印每帧 parse/infer/pack 耗时
=====================================================================
"""
import os
import sys
import time
import threading

import numpy as np
import torch
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField

DEFAULT_CYLINDER3D_ROOT = "/home/heu/deep_learning/lib/cylinder3d"

DEBUG = False


class Cylinder3DInfer:
    """推理引擎。模型工程路径与权重路径都由外部传入。"""

    def __init__(self, root, model_path):
        t0 = time.time()
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise RuntimeError(f"[cylinder3d] 工程根目录不存在: {root}（用 ~cylinder3d_root 指定）")
        os.chdir(root)
        if root not in sys.path:
            sys.path.append(root)

        import yaml
        from builder import model_builder
        from utils.load_save_util import load_checkpoint

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cuda.matmul.allow_tf32 = True

        with open(os.path.join(root, "config/semantickitti.yaml"), 'r', encoding='utf-8') as f:
            configs = yaml.safe_load(f)
        dataset_cfg = configs['dataset_params']
        model_cfg = configs['model_params']

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.grid_size = np.array(model_cfg['output_shape'], dtype=np.float32)
        self.max_bound = np.array(dataset_cfg['max_volume_space'], dtype=np.float32)
        self.min_bound = np.array(dataset_cfg['min_volume_space'], dtype=np.float32)
        self.r_max = float(self.max_bound[0])          # 模型感知半径（50 m）

        import inspect
        if len(inspect.signature(model_builder.build).parameters) >= 2:
            self.model = model_builder.build(model_cfg, init_size=model_cfg['init_size'])
        else:
            self.model = model_builder.build(model_cfg)
        if not model_path:
            model_path = configs['train_params']['model_use_path']
        if not os.path.isabs(model_path):
            model_path = os.path.join(root, model_path)
        if not os.path.exists(model_path):
            raise RuntimeError(f"[cylinder3d] 权重不存在: {model_path}（用 ~model_path 指定）")
        print(f"[INFO] 载入权重: {model_path}")
        self.model = load_checkpoint(model_path, self.model).to(self.device).eval()

        with open(os.path.join(root, dataset_cfg['label_mapping'])) as f:
            learning_map_inv = yaml.safe_load(f)['learning_map_inv']
        self.label_map = np.array([learning_map_inv[i] for i in range(20)], dtype=np.uint32)
        print(f"[INFO] 推理引擎初始化完成，耗时: {(time.time() - t0) * 1000:.2f} ms")

    def _preprocess(self, points):
        xyz = points[:, :3].astype(np.float32)
        intensity = points[:, 3:4].astype(np.float32)
        rho = np.sqrt(xyz[:, 0] ** 2 + xyz[:, 1] ** 2)
        phi = np.arctan2(xyz[:, 1], xyz[:, 0])
        xyz_pol = np.stack([rho, phi, xyz[:, 2]], axis=1).astype(np.float32)

        intervals = (self.max_bound - self.min_bound) / (self.grid_size - 1.0)
        clipped = np.clip(xyz_pol, self.min_bound, self.max_bound)
        grid_ind = np.floor((clipped - self.min_bound) / intervals).astype(np.int64)
        centers = (grid_ind.astype(np.float32) + 0.5) * intervals + self.min_bound
        feat = np.concatenate([xyz_pol - centers, xyz_pol, xyz[:, :2], intensity],
                              axis=1, dtype=np.float32)
        return feat, grid_ind

    @torch.no_grad()
    def infer(self, points_np):
        """@return (labels_raw[原始KITTI标签], confidences)"""
        feat_np, grid_np = self._preprocess(points_np)
        feat_gpu = [torch.from_numpy(feat_np).to(self.device, non_blocking=True)]
        grid_gpu = [torch.from_numpy(grid_np).to(self.device, non_blocking=True)]

        pred_raw = self.model(feat_gpu, grid_gpu, batch_size=1)[0]   # (C, X, Y, Z)
        if self.device.type == 'cuda':
            torch.cuda.synchronize()

        # 只取每条点所在体素的 logits 再 softmax（原脚本对整个 (C,X,Y,Z) 张量做 softmax，纯浪费）
        gx = torch.from_numpy(grid_np[:, 0]).to(self.device)
        gy = torch.from_numpy(grid_np[:, 1]).to(self.device)
        gz = torch.from_numpy(grid_np[:, 2]).to(self.device)
        logits = pred_raw[:, gx, gy, gz]              # (C, N)
        conf, idx = torch.softmax(logits, dim=0).max(dim=0)
        labels = self.label_map[idx.cpu().numpy()]
        return labels, conf.cpu().numpy()


class SemanticROSNode:
    FIELDS = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('ring', 16, PointField.UINT16, 1),
        PointField('time', 18, PointField.FLOAT32, 1),
        PointField('label', 22, PointField.UINT16, 1),   # 原始 SemanticKITTI 标签
        PointField('prob', 24, PointField.UINT8, 1),      # 置信度 * 255
    ]
    POINT_STEP = 25

    def __init__(self):
        rospy.init_node('deros_cylinder3d_semantic', anonymous=False)
        global DEBUG

        self.cloud_topic = rospy.get_param('~cloud_topic', '/kitti/velo/pointcloud')
        self.semantic_topic = rospy.get_param('~semantic_topic', '/kitti/velo/semantic_pointcloud')
        self.keep_out_of_range = bool(rospy.get_param('~keep_out_of_range', True))
        DEBUG = bool(rospy.get_param('~debug', False))

        root = rospy.get_param('~cylinder3d_root', DEFAULT_CYLINDER3D_ROOT)
        model_path = rospy.get_param('~model_path', '')
        self.engine = Cylinder3DInfer(root, model_path)
        self.warmup()

        self.latest_msg = None
        self.has_new_frame = False
        self.lock = threading.Lock()
        self.n_recv = 0          # 收到帧数
        self.n_pub = 0           # 发布帧数
        self.n_overwritten = 0   # 因上一帧还没处理而被覆盖（丢弃）的帧数
        self.t_proc_sum = 0.0    # 累计处理耗时

        self.sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.callback,
                                    queue_size=2, tcp_nodelay=True)
        self.pub = rospy.Publisher(self.semantic_topic, PointCloud2,
                                   queue_size=2, tcp_nodelay=True)

        threading.Thread(target=self.worker_thread, daemon=True).start()
        rospy.loginfo("[online-semantic] Cylinder3D: %s -> %s (keep_out_of_range=%s)",
                      self.cloud_topic, self.semantic_topic, self.keep_out_of_range)

    def warmup(self):
        """用一帧仿真点云先跑一次，把 CUDA/spconv 首次开销吃掉。
        否则数据流开头的头几帧会被"只保留最新帧"的逻辑丢掉，轨迹比 GT 少几行。"""
        try:
            n = 120000
            rng = np.random.default_rng(0)
            r = 45.0 * np.sqrt(rng.random(n))
            th = rng.random(n) * 2 * np.pi
            pts = np.column_stack([r * np.cos(th), r * np.sin(th),
                                   rng.uniform(-2.0, 1.5, n),
                                   rng.uniform(0.0, 1.0, n)]).astype(np.float32)
            t0 = time.time()
            self.engine.infer(pts)
            if self.engine.device.type == 'cuda':
                torch.cuda.synchronize()
            print(f"[INFO] 预热完成，耗时 {(time.time() - t0) * 1000:.0f} ms")
        except Exception as e:  # noqa: BLE001
            rospy.logwarn("预热失败（不影响运行）: %s", e)

    def callback(self, msg):
        with self.lock:
            if self.has_new_frame:          # 上一帧还没处理，被覆盖 = 丢帧
                self.n_overwritten += 1
            self.n_recv += 1
            self.latest_msg = msg
            self.has_new_frame = True

    @staticmethod
    def parse_pc_full(msg):
        """@return xyz(float32 Nx3), intensity, ring, time, valid_mask"""
        arr = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        names = arr.dtype.names
        x = arr['x'].astype(np.float32)
        y = arr['y'].astype(np.float32)
        z = arr['z'].astype(np.float32)
        valid = (z != -100000.0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

        if 'intensity' in names:
            intensity = arr['intensity'].astype(np.float32)
        elif 'i' in names:
            intensity = arr['i'].astype(np.float32)
        else:
            intensity = np.zeros_like(x)

        ring = arr['ring'].astype(np.uint16) if 'ring' in names else np.zeros(len(x), np.uint16)
        tval = arr['time'].astype(np.float32) if 'time' in names else np.zeros(len(x), np.float32)
        return np.column_stack([x, y, z]).astype(np.float32), intensity, ring, tval, valid

    def worker_thread(self):
        while not rospy.is_shutdown():
            msg = None
            with self.lock:
                if self.has_new_frame and self.latest_msg is not None:
                    msg = self.latest_msg
                    self.has_new_frame = False
            if msg is None:
                time.sleep(0.0001)
                continue

            t0 = time.time()
            try:
                self.process(msg)
                self.n_pub += 1
            except Exception as e:  # noqa: BLE001
                import traceback
                rospy.logerr("帧处理异常: %s\n%s", e, traceback.format_exc())
            dt_ms = (time.time() - t0) * 1000
            self.t_proc_sum += dt_ms
            if DEBUG:
                print(f"[DEBUG] 单帧总耗时: {dt_ms:.2f} ms")
            # 每 50 帧报一次真实吞吐与丢帧来源（上游发得快、本节点处理不过来时就会丢在这）
            if self.n_pub % 50 == 0:
                print("[INFO][Cylinder3D] 收到 %d 帧 / 发布 %d 帧（被覆盖丢弃 %d）/ 处理能力 %.1f fps（均值 %.1f ms/帧）"
                      % (self.n_recv, self.n_pub, self.n_overwritten,
                         self.n_pub / max(1e-6, self.t_proc_sum / 1000.0), self.t_proc_sum / self.n_pub),
                      flush=True)

    def process(self, msg):
        t0 = time.time()
        xyz_all, intensity_all, ring_all, time_all, valid = self.parse_pc_full(msg)
        if valid.sum() == 0:
            return
        t_parse = (time.time() - t0) * 1000

        idx = np.nonzero(valid)[0]
        xyz_v = xyz_all[idx]
        int_v = intensity_all[idx]

        # 只对模型感知半径内的点做推理（半径外的点会被裁到边界体素，给出错误标签）
        rho = np.hypot(xyz_v[:, 0], xyz_v[:, 1])
        in_rng = rho <= self.engine.r_max
        keep = np.ones(len(idx), dtype=bool) if self.keep_out_of_range else in_rng

        labels = np.zeros(len(idx), dtype=np.uint16)   # 0 = unlabeled
        probs = np.zeros(len(idx), dtype=np.uint8)

        t1 = time.time()
        sel = np.nonzero(in_rng)[0]
        if len(sel) > 0:
            points = np.column_stack([xyz_v[sel], int_v[sel]])
            pred_labels, conf = self.engine.infer(points)
            labels[sel] = pred_labels.astype(np.uint16)
            probs[sel] = np.clip(conf * 255.0, 0, 255).astype(np.uint8)
        t_infer = (time.time() - t1) * 1000

        t2 = time.time()
        sel_keep = np.nonzero(keep)[0]
        n = len(sel_keep)
        out = np.zeros(n, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('intensity', np.float32), ('ring', np.uint16), ('time', np.float32),
            ('label', np.uint16), ('prob', np.uint8)]))
        out['x'] = xyz_v[sel_keep, 0]
        out['y'] = xyz_v[sel_keep, 1]
        out['z'] = xyz_v[sel_keep, 2]
        out['intensity'] = int_v[sel_keep]
        out['ring'] = ring_all[idx][sel_keep]
        out['time'] = time_all[idx][sel_keep]
        out['label'] = labels[sel_keep]
        out['prob'] = probs[sel_keep]

        cloud = PointCloud2()
        cloud.header = msg.header
        cloud.height = 1
        cloud.width = n
        cloud.fields = self.FIELDS
        cloud.is_bigendian = False
        cloud.point_step = self.POINT_STEP
        cloud.row_step = cloud.point_step * n
        cloud.is_dense = True
        cloud.data = out.tobytes()
        self.pub.publish(cloud)

        if DEBUG:
            print(f"[DEBUG] 单帧: parse={t_parse:.1f} infer={t_infer:.1f} "
                  f"pack+publish={(time.time() - t2) * 1000:.1f} ms | pts_out={n}")


if __name__ == '__main__':
    try:
        SemanticROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
