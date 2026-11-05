#!/bin/bash
# DeRo-SLAM 在线语义推理（SegNet4D，4D 多帧模型）
# Python 环境：/home/heu/application/venvs/segnet4d
#   （由 openpoints 的 python3.8 建的 venv：torch 2.4.1 + spconv-cu120 + pytorch-lightning 1.5.10
#     + torch-geometric/torch-cluster；sptr_cuda、iou3d_nms_cuda、Array_Index 已编译）
# 可用环境变量 DEROS_SEGNET4D_PY 覆盖解释器路径。
# 用法（由 roslaunch 调用时从 rosparam 读配置，也可手动运行）：
#   rosrun deros run_segnet4d.sh _model_path:=/path/to/semantickitti.ckpt..bin
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${DEROS_SEGNET4D_PY:-/home/heu/application/venvs/segnet4d/bin/python}"

# conda/venv 缺 netifaces（rospy 连接对端时要用），用系统 cpython-38 的 .so 兜底
export PYTHONPATH="${HERE}/semantic/pyext:/opt/ros/noetic/lib/python3/dist-packages:${PYTHONPATH}"
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libffi.so.7
exec "$PY" "${HERE}/semantic/segnet4d_ros_node.py" "$@"
