#!/bin/bash
# DeRo-SLAM 在线语义推理（RandLA-Net）
# 移植自 semantic_liosam/src/kitti_odometry/scripts/run_semantic.sh
# 用法（由 roslaunch 调用时从 rosparam 读话题，也可手动运行）：
#   rosrun deros run_randlanet.sh
#   rosrun deros run_randlanet.sh _cloud_topic:=/kitti/velo/pointcloud _semantic_topic:=/kitti/velo/semantic_pointcloud
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# conda env 缺 netifaces（rospy 连接对端时要用），用系统 cpython-38 的 .so 兜底
export PYTHONPATH="${HERE}/semantic/pyext:/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
exec /home/heu/application/Miniconda3/envs/randlanet/bin/python3 \
    "${HERE}/semantic/randlanet_ros_node.py" "$@"
