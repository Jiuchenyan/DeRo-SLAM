#!/bin/bash
# DeRo-SLAM 在线语义推理（Cylinder3D）
# 移植自 semantic_liosam/src/kitti_odometry/scripts/run_cylinder3d.sh
# 用法（由 roslaunch 调用时从 rosparam 读话题，也可手动运行）：
#   rosrun deros run_cylinder3d.sh
#   rosrun deros run_cylinder3d.sh _cloud_topic:=/kitti/velo/pointcloud _semantic_topic:=/kitti/velo/semantic_pointcloud
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# conda env 缺 netifaces（rospy 连接对端时要用），用系统 cpython-38 的 .so 兜底
export PYTHONPATH="${HERE}/semantic/pyext:/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
# 解释器：默认系统 python3，可用 DEROS_CYLINDER3D_PY 指定 conda 环境
PY="${DEROS_CYLINDER3D_PY:-python3}"
exec "$PY" \
    "${HERE}/semantic/cylinder3d_ros_node.py" "$@"
