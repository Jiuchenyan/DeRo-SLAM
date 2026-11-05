#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SegNet4D 在线语义推理节点（第三个后端）
=====================================================================
参考 SegNet4D 官方 scripts/predict_semantickitti.py 的推理流程改写为 ROS 在线节点。

SegNet4D 是 **4D（多帧）** 语义分割：需要当前帧 + 上一帧，并把上一帧按相对位姿对齐到当前帧，
再算 BEV 高度残差作为 motion feature。相对位姿来源（~motion_source）：
  tf   （默认）订阅 /tf，取 <tf_fixed_frame>-><tf_base_frame> 在前后两帧时刻的位姿求相对变换
  none  不做运动补偿（等价于两帧对齐为零位移，车辆移动时精度会下降）

rosparam:
  ~cloud_topic        输入原始点云          (默认 /kitti/velo/pointcloud)
  ~semantic_topic     输出语义点云          (默认 /kitti/velo/semantic_pointcloud)
  ~segnet4d_root      SegNet4D 工程根目录   (默认 ~/deep_learning/lib/SegNet4D)
  ~model_path         权重路径（*必填*，官网下载的 .ckpt/.bin）
  ~motion_source      tf / none
  ~tf_fixed_frame     tf 世界系 (默认 world)
  ~tf_base_frame      tf 车体系 (默认 base_link)
  ~label_head         语义头：semantic(20类,默认) / semantic_all(26类,含动态 252~259) / mos(3类动静)
  ~keep_out_of_range  超出模型点云范围的点是否发布（label=0，保留几何）
  ~debug              打印每帧耗时
=====================================================================
"""
import os
import sys
import threading
import time
from collections import deque

import numpy as np
import rospy
import ros_numpy
import torch
import torch.nn.functional as F
from sensor_msgs.msg import PointCloud2, PointField
from tf2_msgs.msg import TFMessage

DEFAULT_SEGNET4D_ROOT = os.path.expanduser("~/deep_learning/lib/SegNet4D")
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEBUG = False


# ----------------------------------------------------------------------------
# 极简 TF 缓存（只订阅 /tf，不依赖 tf2_ros），用于取 world->base_link 的位姿
# ----------------------------------------------------------------------------
class TfPoseCache:
    def __init__(self, fixed_frame, base_frame, maxlen=400):
        self.fixed = fixed_frame.strip('/')
        self.base = base_frame.strip('/')
        self.data = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self.sub = rospy.Subscriber('/tf', TFMessage, self._cb, queue_size=200, tcp_nodelay=True)

    def _cb(self, msg):
        for tr in msg.transforms:
            if tr.header.frame_id.strip('/') != self.fixed:
                continue
            if tr.child_frame_id.strip('/') != self.base:
                continue
            q = tr.transform.rotation
            t = tr.transform.translation
            x, y, z, w = q.x, q.y, q.z, q.w
            R = np.array([
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ], dtype=np.float64)
            T = np.eye(4)
            T[:3, :3] = R
            T[:3, 3] = [t.x, t.y, t.z]
            with self.lock:
                self.data.append((tr.header.stamp.to_sec(), T))

    def lookup(self, stamp_sec):
        """取 <= stamp 的最近一帧位姿；没有则返回 None"""
        with self.lock:
            best = None
            for s, T in self.data:
                if s <= stamp_sec + 1e-3:
                    best = (s, T)
                else:
                    break
        return None if best is None else best[1]


# ----------------------------------------------------------------------------
# 推理引擎
# ----------------------------------------------------------------------------
class SegNet4DInfer:
    # 头 -> (config 文件键, 语义头名称说明)
    HEADS = {
        'semantic':     'SEMANTIC_CONFIG_FILE',      # 20 类：0,10,11,...,81
        'semantic_all': 'SEMANTIC_CONFIG_FILE_ALL',  # 26 类：额外含 252/253/254/255/258/259 动态类
        'mos':          'SEMANTIC_MOS_CONFIG_FILE',  # 3 类：unlabeled / static / moving
    }

    def __init__(self, root, model_path, label_head='semantic'):
        t0 = time.time()
        root = os.path.abspath(root)
        if not os.path.isdir(root):
            raise RuntimeError(f"[segnet4d] 工程根目录不存在: {root}（用 ~segnet4d_root 指定）")
        if not os.path.isabs(model_path):
            model_path = os.path.join(root, model_path)
        if not os.path.exists(model_path):
            raise RuntimeError(f"[segnet4d] 权重不存在: {model_path}（用 ~model_path 指定）")

        os.chdir(root)
        if root not in sys.path:
            sys.path.insert(0, root)

        import yaml
        import models.models as models
        from models.backbones_3d.voxel_generate import VoxelGenerate
        from spconv.pytorch.utils import gather_features_by_pc_voxel_id
        from dataloader.src import Array_Index

        self.Array_Index = Array_Index
        self.gather = gather_features_by_pc_voxel_id

        ckpt = torch.load(model_path, map_location=DEVICE)
        cfg = ckpt["hyper_parameters"]
        cfg["TRAIN"]["BATCH_SIZE"] = 1
        cfg["DATA"]["ONLINE_TRAIN"] = True
        cfg["DATA"]["TRANSFORM"] = True
        cfg["DATA"]["POSES"] = "poses.txt"
        cfg["DATA"]["POINT_CLOUD_RANGE_MIN"] = [-2.5, -1.5, -3, 2.5, 1.5, 1]

        self.cfg = cfg
        self.pc_range = np.array(cfg["DATA"]["POINT_CLOUD_RANGE"], dtype=np.float32)
        self.pc_range_min = np.array(cfg["DATA"]["POINT_CLOUD_RANGE_MIN"], dtype=np.float32)
        self.grid = np.array(cfg["DATA"]["VOXEL_SIZE"], dtype=np.float32)
        self.n_past = int(cfg["MODEL"]["N_PAST_STEPS"])
        in_channel = len(cfg["MODEL"]["POINT_FEATURE_ENCODING"]["src_feature_list"]) + self.n_past - 1

        # 类别映射：learning 索引 -> 原始 SemanticKITTI 标签
        label_head = label_head if label_head in self.HEADS else 'semantic'
        self.label_head = label_head
        sem_cfg = yaml.safe_load(open(os.path.join(root, cfg["DATA"][self.HEADS[label_head]])))
        self.learning_map_inv = {int(k): int(v) for k, v in sem_cfg["learning_map_inv"].items()}
        mos_cfg = yaml.safe_load(open(os.path.join(root, cfg["DATA"]["SEMANTIC_MOS_CONFIG_FILE"])))
        if label_head == 'mos':
            self.ignore_index = [int(k) for k, v in mos_cfg["learning_ignore"].items() if v]
        else:
            self.ignore_index = [int(k) for k, v in sem_cfg["learning_ignore"].items() if v]

        self.model = models.SegNet4D.load_from_checkpoint(model_path, hparams=cfg, map_location=DEVICE)
        self.model.to(DEVICE).eval()

        self.voxel_generate = VoxelGenerate(self.grid, self.pc_range, 100000, 5, in_channel)
        self.bev_h = int((self.pc_range[3] - self.pc_range[0]) / self.grid[0])
        self.bev_w = int((self.pc_range[4] - self.pc_range[1]) / self.grid[1])
        n_raw = sorted(set(int(v) for v in self.learning_map_inv.values()))
        print(f"[INFO] SegNet4D 载入完成 | n_past={self.n_past} | label_head={label_head} "
              f"({len(self.learning_map_inv)} 类 -> 原始标签 {n_raw[0]}..{n_raw[-1]})")
        print(f"[INFO] voxel_range={self.pc_range.tolist()} | bev={self.bev_h}x{self.bev_w} | "
              f"耗时 {(time.time() - t0) * 1000:.0f} ms")

    # ---------- 工具 ----------
    def _range_mask(self, xyz, lim):
        return ((xyz[:, 0] >= lim[0]) & (xyz[:, 0] < lim[3]) &
                (xyz[:, 1] >= lim[1]) & (xyz[:, 1] < lim[4]) &
                (xyz[:, 2] >= lim[2]) & (xyz[:, 2] < lim[5]))

    def _bev_height(self, xyz):
        img = self.pc_range[2] * np.ones([self.bev_h, self.bev_w], dtype=np.float32)
        mx = self.pc_range[2] * np.ones([self.bev_h, self.bev_w], dtype=np.float32)
        mn = self.pc_range[5] * np.ones([self.bev_h, self.bev_w], dtype=np.float32)
        return self.Array_Index.pointcloud2bevHeightSingleThread_1(
            xyz[:, :3].astype(np.float32), self.pc_range, img, mx, mn, float(self.grid[0]))

    def _motion_feature(self, clouds):
        """clouds: [past..., current]（都已裁剪范围）；返回 (N_cur, n_past-1) 的 BEV 高度残差"""
        cur = clouds[-1]
        feat = np.zeros([cur.shape[0], self.n_past - 1], dtype=np.float16)
        if self.n_past <= 1:
            return feat
        bev_cur = self._bev_height(cur)
        xi = ((-cur[:, 0] - self.pc_range[0]) / self.grid[0]).astype(np.int32)
        yi = ((-cur[:, 1] - self.pc_range[1]) / self.grid[1]).astype(np.int32)
        xi[xi == self.bev_h] = self.bev_h - 1
        yi[yi == self.bev_w] = self.bev_w - 1
        np.clip(xi, 0, self.bev_h - 1, out=xi)
        np.clip(yi, 0, self.bev_w - 1, out=yi)
        for i in range(self.n_past - 1):
            bev_prev = self._bev_height(clouds[self.n_past - i - 2])
            feat[:, i] = (bev_cur - bev_prev)[xi, yi]
        return feat

    @torch.no_grad()
    def infer(self, cur_xyz_i, past_xyz_list):
        """cur_xyz_i: 当前帧 (x,y,z,intensity)；past_xyz_list: 已对齐到当前帧的历史帧（老->新）"""
        # 当前帧范围掩码（和官方脚本一致：先按大范围取，再挖掉近处的小范围）
        mask = self._range_mask(cur_xyz_i, self.pc_range) & ~self._range_mask(cur_xyz_i, self.pc_range_min)
        cur_sel = cur_xyz_i[mask]
        clouds = []
        for p in past_xyz_list[-(self.n_past - 1):]:
            m = self._range_mask(p, self.pc_range) & ~self._range_mask(p, self.pc_range_min)
            clouds.append(p[m])
        while len(clouds) < self.n_past - 1:      # 帧数不够时用当前帧补
            clouds.insert(0, cur_sel)
        clouds.append(cur_sel)

        motion = self._motion_feature(clouds).astype(np.float32)
        feat = np.hstack([cur_sel, motion])
        tensor = torch.tensor(feat, dtype=torch.float32, device=DEVICE)

        data_dict = {
            "meta": (0, 0, None),
            "current_point_with_feature_tensor": [tensor],
            "batch_size_npast": self.n_past,
            "range_mask": [mask],
            "batch_size": 1,
        }
        data_dict = self.voxel_generate(data_dict)
        mos_feature, semantic_feature, semantic_all_feature, _, _ = self.model.forward(data_dict, 'test')
        feat_map = {'semantic': semantic_feature,
                    'semantic_all': semantic_all_feature,
                    'mos': mos_feature}
        semantic_feature = feat_map[self.label_head]

        batch_mask = data_dict['voxel_coords'][:, 0] == 0
        feats = self.gather(semantic_feature[batch_mask], data_dict['pc_voxel_id'][0])
        feats[:, self.ignore_index] = -float("inf")
        prob = F.softmax(feats, dim=1)
        conf, idx = prob.max(dim=1)
        labels = idx.cpu().numpy()

        raw = np.array([self.learning_map_inv.get(int(l), 0) for l in labels], dtype=np.int32)
        return mask, raw, conf.cpu().numpy()


# ----------------------------------------------------------------------------
# ROS 节点
# ----------------------------------------------------------------------------
class SemanticROSNode:
    FIELDS = [
        PointField('x', 0, PointField.FLOAT32, 1),
        PointField('y', 4, PointField.FLOAT32, 1),
        PointField('z', 8, PointField.FLOAT32, 1),
        PointField('intensity', 12, PointField.FLOAT32, 1),
        PointField('ring', 16, PointField.UINT16, 1),
        PointField('time', 18, PointField.FLOAT32, 1),
        PointField('label', 22, PointField.UINT16, 1),
        PointField('prob', 24, PointField.UINT8, 1),
    ]
    POINT_STEP = 25

    def __init__(self):
        rospy.init_node('deros_segnet4d_semantic', anonymous=False)
        global DEBUG
        self.cloud_topic = rospy.get_param('~cloud_topic', '/kitti/velo/pointcloud')
        self.semantic_topic = rospy.get_param('~semantic_topic', '/kitti/velo/semantic_pointcloud')
        self.keep_out_of_range = bool(rospy.get_param('~keep_out_of_range', True))
        DEBUG = bool(rospy.get_param('~debug', False))
        self.motion_source = str(rospy.get_param('~motion_source', 'tf')).lower()
        fixed = rospy.get_param('~tf_fixed_frame', 'world')
        base = rospy.get_param('~tf_base_frame', 'base_link')

        root = rospy.get_param('~segnet4d_root', DEFAULT_SEGNET4D_ROOT)
        model_path = rospy.get_param('~model_path', '')
        label_head = str(rospy.get_param('~label_head', 'semantic')).lower()
        self.engine = SegNet4DInfer(root, model_path, label_head)
        self.warmup()

        self.tf_cache = TfPoseCache(fixed, base) if self.motion_source == 'tf' else None
        self.history = deque(maxlen=max(1, self.engine.n_past - 1))   # (stamp, xyz_i)
        self.tf_warned = False

        self.latest_msg = None
        self.has_new_frame = False
        self.lock = threading.Lock()
        self.n_recv = 0          # 收到帧数
        self.n_pub = 0           # 发布帧数
        self.n_overwritten = 0   # 因上一帧还没处理而被覆盖（丢弃）的帧数
        self.t_proc_sum = 0.0    # 累计处理耗时

        self.sub = rospy.Subscriber(self.cloud_topic, PointCloud2, self.callback,
                                    queue_size=2, tcp_nodelay=True)
        self.pub = rospy.Publisher(self.semantic_topic, PointCloud2, queue_size=2, tcp_nodelay=True)
        threading.Thread(target=self.worker_thread, daemon=True).start()
        rospy.loginfo("[online-semantic] SegNet4D: %s -> %s (n_past=%d, motion_source=%s)",
                      self.cloud_topic, self.semantic_topic, self.engine.n_past, self.motion_source)

    def warmup(self):
        """用两帧仿真点云先跑一次，吃掉 CUDA/spconv 首次开销"""
        try:
            n = 120000
            rng = np.random.default_rng(0)
            r = 45.0 * np.sqrt(rng.random(n))
            th = rng.random(n) * 2 * np.pi
            cur = np.column_stack([r * np.cos(th), r * np.sin(th),
                                   rng.uniform(-2.0, 1.5, n), rng.uniform(0, 1, n)]).astype(np.float32)
            t0 = time.time()
            self.engine.infer(cur, [cur.copy() for _ in range(max(1, self.engine.n_past - 1))])
            if DEVICE.type == 'cuda':
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
        arr = ros_numpy.point_cloud2.pointcloud2_to_array(msg)
        names = arr.dtype.names
        x = arr['x'].astype(np.float32)
        y = arr['y'].astype(np.float32)
        z = arr['z'].astype(np.float32)
        if 'intensity' in names:
            inten = arr['intensity'].astype(np.float32)
        elif 'i' in names:
            inten = arr['i'].astype(np.float32)
        else:
            inten = np.zeros_like(x)
        ring = arr['ring'].astype(np.uint16) if 'ring' in names else np.zeros(len(x), np.uint16)
        tval = arr['time'].astype(np.float32) if 'time' in names else np.zeros(len(x), np.float32)
        valid = (z != -100000.0) & np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
        xyz_i = np.column_stack([x, y, z, inten]).astype(np.float32)
        return xyz_i, ring, tval, valid

    def _align_past(self, stamp_cur, past_list):
        """把历史帧按相对位姿对齐到当前帧坐标系"""
        if self.motion_source != 'tf' or self.tf_cache is None:
            return past_list
        T_cur = self.tf_cache.lookup(stamp_cur)
        out = []
        for stamp_p, xyz in past_list:
            T_p = self.tf_cache.lookup(stamp_p)
            if T_cur is None or T_p is None:
                if not self.tf_warned:
                    rospy.logwarn("[segnet4d] TF %s->%s 缺失，motion_source 退化为 none",
                                  self.tf_cache.fixed, self.tf_cache.base)
                    self.tf_warned = True
                out.append(xyz)
                continue
            T_rel = np.linalg.inv(T_cur) @ T_p
            p = xyz[:, :3] @ T_rel[:3, :3].T + T_rel[:3, 3]
            out.append(np.hstack([p, xyz[:, 3:]]).astype(np.float32))
        return out

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
                print("[INFO][SegNet4D] 收到 %d 帧 / 发布 %d 帧（被覆盖丢弃 %d）/ 处理能力 %.1f fps（均值 %.1f ms/帧）"
                      % (self.n_recv, self.n_pub, self.n_overwritten,
                         self.n_pub / max(1e-6, self.t_proc_sum / 1000.0), self.t_proc_sum / self.n_pub),
                      flush=True)

    def process(self, msg):
        xyz_i, ring, tval, valid = self.parse_pc_full(msg)
        if valid.sum() == 0:
            return
        stamp = msg.header.stamp.to_sec()

        past_aligned = self._align_past(stamp, list(self.history))
        while len(past_aligned) < self.engine.n_past - 1:
            past_aligned.insert(0, xyz_i.copy())
        past_aligned = past_aligned[-(self.engine.n_past - 1):]

        mask, labels_raw, conf = self.engine.infer(xyz_i[valid], past_aligned)
        # 把预测写回“通过 valid + 范围掩码”的点
        idx_valid = np.nonzero(valid)[0]
        in_rng = np.zeros(len(xyz_i), dtype=bool)
        in_rng[idx_valid[mask]] = True
        labels = np.zeros(len(xyz_i), dtype=np.uint16)
        probs = np.zeros(len(xyz_i), dtype=np.uint8)
        labels[in_rng] = labels_raw.astype(np.uint16)
        probs[in_rng] = np.clip(conf * 255.0, 0, 255).astype(np.uint8)

        keep = valid if self.keep_out_of_range else in_rng
        sel = np.nonzero(keep)[0]
        n = len(sel)
        out = np.zeros(n, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('intensity', np.float32), ('ring', np.uint16), ('time', np.float32),
            ('label', np.uint16), ('prob', np.uint8)]))
        out['x'] = xyz_i[sel, 0]
        out['y'] = xyz_i[sel, 1]
        out['z'] = xyz_i[sel, 2]
        out['intensity'] = xyz_i[sel, 3]
        out['ring'] = ring[sel]
        out['time'] = tval[sel]
        out['label'] = labels[sel]
        out['prob'] = probs[sel]

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

        # 保存到历史（原始坐标系，用于下一帧对齐）
        self.history.append((stamp, xyz_i[valid]))
        if DEBUG:
            print(f"[DEBUG] 单帧: 点=%d 有效预测=%d 输出=%d" % (len(xyz_i), int(mask.sum()), n))


if __name__ == '__main__':
    try:
        SemanticROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
