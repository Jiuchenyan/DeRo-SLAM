# DeRo-SLAM 在线语义点云（run_mode = 3）

把 `semantic_liosam` 的两个语义推理脚本（`run_randlanet.sh` / `run_cylinder3d.sh`）移植进 DeRo-SLAM，
并拆成两个互相独立的层，由 `launch/online/` 下的文件统一管理。

## 1. 目录结构

```
launch/
├── datasets/                 # 各数据集：核心接收层（run_mode 0/1/2/3 通用）
│   ├── kitti_start.launch    kitti360_start.launch   gode_start.launch
│   ├── mulran_start.launch   mid360.launch
│   └── our_dataset_start.launch  subt_start.launch
└── online/                   # 在线相关
    ├── online_input.launch     # 总入口：参数校验 + include 推理层 + 可选播包
    │                           #   （只在 run_mode==3 时被 datasets 的 launch include）
    └── semantic_inference.launch  # 语义推理节点（话题/模型全参数化，可单独启动）

scripts/
├── eval/                     # 评测与画图脚本（detal_eval_*.py / eval_traj_utils.py / err_traj_plot_*.py）
├── semantic/                 # 在线推理：python 节点 + pyext（netifaces 兜底）+ README
├── run_cylinder3d.sh         # ROS 节点入口（必须在 scripts/ 根：roslaunch 不支持子路径）
└── run_randlanet.sh
```

## 2. 运行模式（原 debug_mode 已改名 run_mode）

| run_mode | 名称 | 说明 |
| --- | --- | --- |
| 0 | ROS_RUN | 纯 ROS 运行：前端按 `ratehz` 节拍，不等后端 |
| 1 | EVAL_ONLY | 只跑评估：不发布任何 ROS 输出（最快） |
| 2 | EVAL_AND_ROS | 评估 + ROS：前端逐帧等后端处理完（默认） |
| 3 | ONLINE_SEMANTIC | 在线：点云从语义点云话题订阅，替代离线 `LoadData` |

**所有模式**在结束时都会：保存里程计轨迹 + 优化轨迹 → 发布（有订阅者）并落盘全局地图 → 自动执行
`eval_cmd` 把指标打印到终端。三种模式（0/1/2）跑完自动结束；模式 3 靠 `online_idle_timeout`
判断数据流结束（见 §5）后同样自动收尾，**不需要 Ctrl-C**。

> 旧参数名 `debug_mode` 节点仍接受（会 warn），launch 已统一为 `run_mode`。

## 3. 数据流

```
rosbag play ──> /kitti/velo/pointcloud
                    │
                    ├─> semantic_inference.launch ──> /kitti/velo/semantic_pointcloud
                    │      （run_mode=3 时由 online_input.launch 自动 include）
                    │
                    └─> kitti_start.launch run_mode:=3 semantic_cloud_topic:=<同一话题>
                            = ParseCloud2Msg() 代替 LoadData()，前端/后端/回环/PGO 不变
```

输出点云字段：`x,y,z,intensity(F32) ring(U16) time(F32) label(U16) prob(U8)`，`point_step = 25`。
`label` 是原始 SemanticKITTI 标签（0,10,…,259），DeRo 的 `label_map` 直接消费。

## 4. 用法

一条 launch 全搞定（核心 + 推理 + 播包）：

```bash
roslaunch deros kitti_start.launch sequence:=00 run_mode:=3 \
    semantic_cloud_topic:=/kitti/velo/semantic_pointcloud \
    launch_bag:=true
# 想换推理网络：semantic_node:=randlanet
# 想换权重：    cylinder3d_model:=model_save_dir/xxx.pt
```

分层启动（推理在别的机器/别的终端跑时用）：

```bash
# 只起推理层
roslaunch deros semantic_inference.launch \
    cloud_topic:=/kitti/velo/pointcloud semantic_topic:=/kitti/velo/semantic_pointcloud
# 核心只订阅，不重复起推理、不播包
roslaunch deros kitti_start.launch sequence:=00 run_mode:=3 \
    semantic_cloud_topic:=/kitti/velo/semantic_pointcloud \
    launch_semantic:=false
# 自己喂数据
rosbag play --topics /kitti/velo/pointcloud kitti00.bag
```

参数校验（`run_mode:=3` 但忘了给话题）：

```
RLException: while processing .../online/online_input.launch:
Invalid <arg> tag: [run_mode=3 在线语义模式] 必须指定 semantic_cloud_topic（不能为空）...
```

结果文件（与离线模式同名，**后跑的覆盖先跑的**，需要对比就改 `save_path`）：

```
save/evo_log/<data_set>_odometry_<seq>.txt      # 前端里程计
save/evo_log/<data_set>_mapping_<seq>.txt       # 后端优化位姿（评测用它）
save/map/global_map.ply + global_labels.txt     # 全局地图
```

## 5. 参数

### 数据集 launch（以 kitti_start.launch 为例，其它数据集同构）

| arg | 默认 | 说明 |
| --- | --- | --- |
| `run_mode` | `2` | 见 §2 |
| `semantic_cloud_topic` | `""` | **run_mode=3 必填**，等于推理层输出话题 |
| `online_queue_size` | `30` | 订阅缓冲，处理不过来时丢最旧帧 |
| `online_idle_timeout` | `5.0` | 在线：多少秒没有新点云就视为数据流结束 → 自动收尾；`0` = 一直等 |
| `launch_semantic` | **跟随 `launch_bag`** | 是否由本 launch 起推理节点。默认 `$(eval arg('launch_bag'))`：要播包就顺便起推理；"播包但推理在别处跑"时显式给 `false` |
| `semantic_node` | `cylinder3d` | `cylinder3d` / `randlanet` / `segnet4d` |
| `cylinder3d_model` | `model_save_dir/assistant16_63.18.pt` | 相对 `semantic_liosam/src/cylinder3d` |
| `keep_out_of_range` | `true` | 超出模型 50 m 感知半径的点是否发布（label=0，保留几何） |
| `lidar_topic` | 数据集自带 | 原始点云话题（在线时给推理层和播包用） |
| `launch_bag` | `false` | 是否随本 launch 播 rosbag（只播 `lidar_topic`） |
| `bag_file` | 数据集自带 | 包路径，可覆盖 |
| `bag_rate` / `bag_delay` | `1.0` / `0.0` | 播放速率 / 启动后延迟（`0` 且本层起推理时自动等 40 s 让模型加载完；**如果显式给了较小的值，模型还没加载完就会开始播，开头若干帧会丢**） |
| `save_path` | `$(find deros)/save/` | 结果与地图输出目录（结尾要带 `/`） |
| `eval_cmd` | 数据集自带 | 结束时自动执行的评测命令，可覆盖 |
| `visualize` / `recordbag` | `true` / `false` | rviz / 录包（run_mode=1 时不启动） |

### online/semantic_inference.launch —— **模型配置的唯一入口**

| arg | 默认 | 说明 |
| --- | --- | --- |
| `cloud_topic` | `/kitti/velo/pointcloud` | 输入原始点云 |
| `semantic_topic` | `/kitti/velo/semantic_pointcloud` | 输出语义点云 |
| `semantic_node` | `cylinder3d` | `cylinder3d` / `randlanet` / `segnet4d` |
| **`model_path`** | `""` | **通用权重路径**（绝对路径，或相对该后端工程根目录）。空 = 用后端默认权重 |
| `cylinder3d_root` | `/home/heu/ros/projs/semantic_liosam/src/cylinder3d` | Cylinder3D 工程根目录 |
| `cylinder3d_default_model` | `model_save_dir/assistant16_63.18.pt` | Cylinder3D 默认权重 |
| `randlanet_root` | `/home/heu/deep_learning/lib/RandLA-Net-pytorch-main` | RandLA-Net 工程根目录 |
| `randlanet_default_model` | `$(arg randlanet_root)/pretrain_model/checkpoint.tar` | RandLA-Net 默认权重 |
| `segnet4d_root` | `/home/heu/deep_learning/lib/SegNet4D` | SegNet4D 工程根目录 |
| `segnet4d_default_model` | `checkpoint/semantickitti.ckpt..bin` | SegNet4D 默认权重 |
| `motion_source` | `tf` | **SegNet4D 专用**：相对位姿来源 `tf` / `none` |
| `label_head` | `semantic` | **SegNet4D 专用**：`semantic`(20类) / `semantic_all`(**26类，含 252~259 动态**) / `mos`(3类动静) |
| `tf_fixed_frame` / `tf_base_frame` | `world` / `base_link` | TF 坐标系 |
| `keep_out_of_range` | `true` | 超出模型 50 m 感知半径的点是否发布（label=0，保留几何） |
| `debug` | `false` | 打印每帧 parse/infer/pack 耗时 |

节点侧对应 rosparam：`~model_path`、`~cylinder3d_root`、`~randlanet_root`、`~keep_out_of_range`、`~debug`
（**py 里不再硬编码任何工程路径 / 权重路径**，两个后端都是启动时打印实际解析出的路径）。

## 6. 注意事项（实测结论）

1. **权重选择（重要）**
   `config/semantickitti.yaml` 默认的 `model_stu_save_best.pt` 在 KITTI00 第 1 帧上与 GT 一致率只有 **0.18**；
   `assistant16_63.18.pt` 是 **0.90**。launch 默认用后者。

2. **在线模式的结束判定**
   `online_idle_timeout`（默认 5 s）：收到过点云之后，若连续 5 s 没有新点云就认为数据流结束，
   打印 `[online] 数据流结束，共处理 N 帧` 并走收尾（存轨迹 → 全局地图 → 评测）。
   真机长期运行时设 `online_idle_timeout:=0` 让它一直等。
   **不要用 SIGTERM/kill -9 收尾**，那样会跳过收尾三步；用 Ctrl-C 或让它自己 idle 结束。

3. **推理节点冷启动预热**
   Cylinder3D / RandLA-Net 节点启动时都会用一帧 12 万点仿真点云先跑一次（`预热完成，耗时 0.4~0.6 s`），
   吃掉 CUDA/spconv/FAISS 首次开销。否则数据流开头的 2~3 帧会因"只处理最新帧"被丢掉，
   轨迹比 GT 少几行，评测直接报长度不一致。

3.1 **结果目录自动创建**
   `evo_log/`、`map/` 不存在时节点会自己 `create_directories`；否则轨迹会静默写不出来，
   评测随后报 `FileNotFoundError: .../evo_log/xxx_mapping_00.txt`。

4. **实时性（实测分解）**

   KITTI00，12.4 万点/帧，`bag 1.0x` 全链路 **9.6 fps 输入 / 9.6 fps 输出，队列 0、零丢帧**：

   | 阶段 | 耗时 |
   | --- | --- |
   | 语义节点 解析 / 推理 / 打包发布 | 1.0 / ~45 / 2.3 ms |
   | DeRo 订阅+解析 | 0.9 ms |
   | DeRo 前端 `mainProcess` | 14.6 ms（10~40 波动） |
   | DeRo 后端 mapping | 0.3 ms（后端本身 6.4 ms） |

   每帧打印：`[online] idx:235 | wait:90.4 | parse:0.8 | main:9.9 | sync:0.2 | pub:0.1 | all:101.4 ms | queue:0`

   **怎么读这行**（重要，别把 `all` 当算力）：
   - `all = wait + parse + main + sync + pub`；真正的计算只有 `parse+main+sync+pub`（≈16~80 ms）。
   - `wait` 是**等下一帧到达**的时间，它由上游发布速率决定：`wait≈150 ms` 表示上游只有 ~6.7 fps，
     不是 DeRo 慢了。
   - `queue:0` 只说明 **DeRo 自己**没积压；如果上游节点算不过来，帧已经在**上游**被丢了，
     DeRo 自然看不到丢帧。所以判断“是不是真的实时”，要看下面的两处统计，而不是 `queue`。

   **两处新统计**：
   - 语义节点每 50 帧打印一次真实吞吐与丢帧来源：
     `[INFO][SegNet4D] 收到 144 帧 / 发布 100 帧（被覆盖丢弃 43）/ 处理能力 6.7 fps（均值 148.2 ms/帧）`
   - DeRo 结束打印整体速率与算力占比：
     `[online] 数据流结束：处理 N 帧 / 用时 T s = X fps`、
     `[online] 实际计算负载：wait 累计 …s（xx%）→ 真实处理能力 ≈ Y fps`、
     以及 `零丢帧` / `订阅队列因处理不过来丢弃了 K 帧`。

   **所以**：`bag_rate:=1.0` 才是“9.65 Hz 真机节奏”。这时如果语义节点够快（Cylinder3D ~45 ms、
   RandLA-Net ~35 ms，都 >10 fps）就是真实时；SegNet4D ~150 ms 只有 6.7 fps，
   1.0x 下每秒会丢 3 帧左右（丢在推理节点），用 `bag_rate≈0.6` 才是“不丢帧”的等价展示。

   > 已修的坑：节点原本没有异步 spinner，主循环阻塞等点云时回调只能靠 `spinOnce()` 每帧拉一次，
   > 订阅队列不停丢消息 → "一卡一卡、3 fps、退化频发"。现在 `run_mode=3` 时 `main()` 起 `ros::AsyncSpinner(2)`。
   > 另：原 cylinder3d 脚本对整个 `(20,320,240,16)`≈2450 万元素做 softmax 再按体素取值，
   > 移植版先取 `(20,N)` logits 再 softmax，省约 5 ms。

5. **conda 环境缺 netifaces**
   推理脚本用 `LD_PRELOAD=libffi.so.7` + conda python；conda 环境没有 `netifaces`，rospy 连接对端会崩。
   `scripts/semantic/pyext/` 软链了系统 `netifaces.cpython-38-*.so`，启动脚本已加进 `PYTHONPATH`。

6. **管道等价性与精度**
   把 bag 里带 GT label 的点云原样转发到语义话题（不经模型），在线与离线第 1~95 帧位姿逐元素差 ≈1e-8。
   同一段 KITTI00（前 1247 帧，906.3 m）对比：

   | 喂法 | 帧数 | 轨迹长 | ATE rmse |
   | --- | --- | --- | --- |
   | 离线 `LoadData` + 数据集 GT 标签 | 1247 | 906.3 m | **0.609 m** |
   | 在线 1.0x + Cylinder3D 标签 | 1247 | 906.3 m | **0.930 m** |

7. **mid360 没有 rosbag**（数据是 bin/label 文件），`bag_file` 默认为空，`launch_bag` 自动失效。

8. **三个推理后端都可用（实测，KITTI00 前 250~260 帧）**

   | 后端 | 类型 | 环境 | 依赖 | 单帧耗时 | 实测 |
   | --- | --- | --- | --- | --- | --- |
   | Cylinder3D | 单帧 | conda `cylinder3d` | `semantic_liosam/src/cylinder3d` + `model_save_dir/assistant16_63.18.pt` | ~45 ms | ATE 0.2206 m |
   | RandLA-Net | 单帧 | conda `randlanet` | `~/deep_learning/lib/RandLA-Net-pytorch-main`（含 `pretrain_model/checkpoint.tar`） | ~35 ms | ATE 0.2219 m |
   | SegNet4D | **4D（8 帧）** | venv `/home/heu/application/venvs/segnet4d` | `~/deep_learning/lib/SegNet4D`（含 `checkpoint/semantickitti.ckpt..bin`） | ~155 ms | ATE 0.2199 m |

   排查用：`rosrun deros run_randlanet.sh _debug:=true`（或 `run_cylinder3d.sh` / `run_segnet4d.sh`）
   单独起节点，看是否打印加载完成与 `预热完成`。

9. **SegNet4D 环境与源码补丁（已在本机装好，换机器时照做）**

   SegNet4D 官方要求 conda + CUDA11.8，本机 `conda.anaconda.org` 不通，因此用现有 py3.8 环境建了 **venv**：

   ```bash
   # 1) 用 openpoints 的 python3.8（已带 torch 2.4.1）建 venv
   /home/heu/application/Miniconda3/envs/openpoints/bin/python -m venv --system-site-packages \
       /home/heu/application/venvs/segnet4d
   PY=/home/heu/application/venvs/segnet4d/bin/python

   # 2) 装依赖（spconv 用 cu120 版本对应 torch 2.4.1+cu121）
   $PY -m pip install spconv-cu120 "pytorch-lightning==1.5.10" easydict pyyaml timm==0.9.7 pybind11
   $PY -m pip install "torch-geometric==2.5.3"
   $PY -m pip install torch-cluster torch-scatter -f https://data.pyg.org/whl/torch-2.4.1+cu121.html
   $PY -m pip install pycryptodomex python-gnupg     # rosbag 需要

   # 3) 编译 3 个扩展（cpython-38）
   cd ~/deep_learning/lib/SegNet4D/utils/src && mkdir -p build && cd build
   cmake .. -DPython3_EXECUTABLE=$PY && make -j8
   cp Array_Index.cpython-38-x86_64-linux-gnu.so ../../../dataloader/src/
   cp Array_Index.cpython-38-x86_64-linux-gnu.so ../../../models/utils/

   cd ~/deep_learning/lib/SegNet4D && TORCH_CUDA_ARCH_LIST=7.5 $PY setup.py build_ext --inplace   # iou3d_nms_cuda
   cd models/SparseTransformer && TORCH_CUDA_ARCH_LIST=7.5 $PY setup.py build_ext --inplace        # sptr_cuda
   cp sptr_cuda.cpython-38-x86_64-linux-gnu.so $($PY -c "import site;print(site.getsitepackages()[0])")/
   ```

   需要改的两处源码（已改，原文件留有 `.bak`）：
   - `models/SparseTransformer/src/sptr/**/*.cpp`：`#include <THC/THC.h>` → `#include <ATen/ATen.h>`（新 torch 已删 THC）
   - `models/SparseTransformer/sptr/utils.py`：`voxel_grid(pos, batch, size, ...)` → `voxel_grid(pos, size, batch=batch, ...)`
     （torch_geometric 新版把 `size` 挪到第 2 个参数）

10. **SegNet4D 是 4D 模型，注意三点**

   - checkpoint 里 `N_PAST_STEPS=8`（不是 config 文件的 2）：需要当前帧 + 之前 7 帧，节点会自动缓存历史帧；
     流刚开始不足 7 帧时用当前帧补齐。
   - 历史帧要按**相对位姿**对齐到当前帧：`motion_source:=tf` 时订阅 `/tf` 取 `world->base_link`；
     所以 rosbag 播放时必须把 `/tf` 一起播（`online_input.launch` 默认已加 `--topics <lidar> /tf /tf_static`）。
     没有 TF 时会退化成 `none`（不补偿运动），日志会 warn 一次。
   - 速度 ~150 ms/帧（2080Ti），约 6.7 fps：`bag_rate:=1.0`（9.65 Hz）时每秒丢约 3 帧（丢在推理节点内部，
     日志 `[INFO][SegNet4D] 收到 X 帧 / 发布 Y 帧（被覆盖丢弃 Z）` 可查）；要不丢帧就 `bag_rate:=0.6`。

11. **SegNet4D 的动态点（>=250 标签）**

   模型有三个输出头，用 `label_head` 选：

   | label_head | 类别数 | 输出标签 | 用途 |
   | --- | --- | --- | --- |
   | `semantic`（默认） | 20 | 0,10,11,…,81 | 静态语义，与另两个后端同口径 |
   | `semantic_all` | 26 | 上面 20 类 + **252 移动车 / 253 骑行 / 254 行人 / 255 骑摩托 / 258 移动卡车 / 259 移动其他车** | 动态点 |
   | `mos` | 3 | 0 / 9 / 251（unlabeled / static / moving） | 只要动静二分类 |

   DeRo 侧已经接好：`label_map` 把 252~259 映射成内部 20~25，`frontend/Preprocessing.cpp`
   里当一帧出现 `label>=252` 就会把动态点排除在配准/地图之外（代码注释写的正是 “from the segnet4d”）。
   所以想用动态信息：`label_head:=semantic_all` + `remove_foreground:=true`。

   实测（KITTI00，住宅区，动态点少）：
   ```
   帧5  动态点 74 个: {254: 2, 255: 72}
   帧20 动态点 45 个: {253: 2, 255: 43}
   帧40 动态点  9 个: {253: 8, 259: 1}
   ```

## 7. 与非 KITTI 数据集一起用

推理模型是 SemanticKITTI 上训的，直接用在 SubT/MulRan/自采数据上标签不可信，但管道仍可用：

```bash
roslaunch deros subt_start.launch sequence:=long_Corridor run_mode:=3 \
    semantic_cloud_topic:=/velodyne_points_semantic launch_bag:=true
```
