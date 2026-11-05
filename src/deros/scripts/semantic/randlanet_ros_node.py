#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RandLA-Net 在线语义推理节点（在线模式专用）
=====================================================================
移植自：semantic_liosam/src/kitti_odometry/scripts/randlanet_ros.py
改动点：
  1. 输入不再依赖 liosam 的自定义消息 cloud_info，改为普通 sensor_msgs/PointCloud2，
     输出也是普通 PointCloud2（因此 DeRo-SLAM 侧无需编译 kitti_odometry 的消息）；
  2. 输出字段：
       x, y, z, intensity, ring, time, label(UINT16 原始 SemanticKITTI 标签), prob(UINT8)
     RandLA-Net 的类别索引 1..19 在这里映射回原始 SemanticKITTI 标签
     (10,11,15,18,20,30,31,32,40,44,48,49,50,51,70,71,72,80,81)，
     DeRo-SLAM 的 label_map 直接消费；
  3. 推理引擎 randlanet_infer.py 与工程内同目录副本保持一致（模型代码未改）。

rosparam:
  ~cloud_topic    (string, "/kitti/velo/pointcloud")
  ~semantic_topic (string, "/kitti/velo/semantic_pointcloud")
  ~debug          (bool, false)
"""
import os
import sys
import time
import threading

import numpy as np
import rospy
import ros_numpy
from sensor_msgs.msg import PointCloud2, PointField

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import randlanet_infer
from randlanet_infer import RandLANetPreciseInfer, DEFAULT_RANDLANET_ROOT  # noqa: E402

DEBUG = False

# RandLA-Net 输出索引(1..19) -> 原始 SemanticKITTI 标签
IDX_TO_KITTI = np.zeros(21, dtype=np.uint16)
for _idx, _raw in {
    1: 10, 2: 11, 3: 15, 4: 18, 5: 20,
    6: 30, 7: 31, 8: 32,
    9: 40, 10: 44, 11: 48, 12: 49,
    13: 50, 14: 51,
    15: 70, 16: 71, 17: 72,
    18: 80, 19: 81,
}.items():
    IDX_TO_KITTI[_idx] = _raw


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
        rospy.init_node('deros_randlanet_semantic', anonymous=False)
        global DEBUG
        self.cloud_topic = rospy.get_param('~cloud_topic', '/kitti/velo/pointcloud')
        self.semantic_topic = rospy.get_param('~semantic_topic', '/kitti/velo/semantic_pointcloud')
        DEBUG = bool(rospy.get_param('~debug', False))

        self.model = None
        root = rospy.get_param('~randlanet_root', DEFAULT_RANDLANET_ROOT)
        model_path = rospy.get_param('~model_path', '')
        self.model = RandLANetPreciseInfer(root, model_path)
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
        rospy.loginfo("[online-semantic] %s -> %s", self.cloud_topic, self.semantic_topic)

    def warmup(self):
        """用一帧仿真点云先跑一次，把 FAISS/CUDA 首次开销吃掉；
        否则数据流开头的几帧会因“只处理最新帧”而被丢掉，轨迹比 GT 少几行。"""
        try:
            n = 120000
            rng = np.random.default_rng(0)
            r = 45.0 * np.sqrt(rng.random(n))
            th = rng.random(n) * 2 * np.pi
            pts = np.column_stack([r * np.cos(th), r * np.sin(th),
                                   rng.uniform(-2.0, 1.5, n)]).astype(np.float32)
            t0 = time.time()
            self.model.infer(pts)
            print(f"[INFO] 预热完成，耗时 {(time.time() - t0) * 1000:.0f} ms")
        except Exception as e:  # noqa: BLE001
            print(f"[WARN] 预热失败（不影响运行）: {e}")

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
                print("[INFO][RandLA-Net] 收到 %d 帧 / 发布 %d 帧（被覆盖丢弃 %d）/ 处理能力 %.1f fps（均值 %.1f ms/帧）"
                      % (self.n_recv, self.n_pub, self.n_overwritten,
                         self.n_pub / max(1e-6, self.t_proc_sum / 1000.0), self.t_proc_sum / self.n_pub),
                      flush=True)

    def process(self, msg):
        xyz_all, intensity_all, ring_all, time_all, valid = self.parse_pc_full(msg)
        if valid.sum() == 0:
            return

        idx = np.nonzero(valid)[0]
        xyz_v = xyz_all[idx]
        int_v = intensity_all[idx]

        valid_xyz, idx_labels, conf = self.model.infer(xyz_v)
        if idx_labels is None:
            rospy.logwarn("推理失败，跳过此帧")
            return

        labels = IDX_TO_KITTI[np.clip(idx_labels.astype(np.int32), 0, 20)]

        n = len(valid_xyz)
        out = np.zeros(n, dtype=np.dtype([
            ('x', np.float32), ('y', np.float32), ('z', np.float32),
            ('intensity', np.float32), ('ring', np.uint16), ('time', np.float32),
            ('label', np.uint16), ('prob', np.uint8)]))
        out['x'] = valid_xyz[:, 0]
        out['y'] = valid_xyz[:, 1]
        out['z'] = valid_xyz[:, 2]
        out['intensity'] = int_v
        out['ring'] = ring_all[idx]
        out['time'] = time_all[idx]
        out['label'] = labels[:n]
        out['prob'] = np.clip(conf[:n] * 255.0, 0, 255).astype(np.uint8)

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


if __name__ == '__main__':
    try:
        SemanticROSNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
