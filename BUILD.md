# Build & Run

This repository is the **release package** of DeRo-SLAM. The algorithm sources
are not included: they ship precompiled as
`src/deros_core/lib/libderos_core.so` together with the public headers in
`src/deros_core/include`. `src/deros` is the ROS application layer and can be
read and modified freely.

## Requirements

- Ubuntu 20.04 + ROS Noetic
- OpenCV 4, TBB, glog, OpenBLAS and Suitesparse (the shared libraries
  `libderos_core.so` links against):

```bash
sudo apt install libgoogle-glog-dev libsuitesparse-dev libopenblas-dev \
                 libtbb-dev libopencv-dev
```

- GTSAM 4.x (`libgtsam.so.4`), built from source or from your distribution
- Eigen and Sophus headers are vendored under
  `src/deros_core/include/third_party` — nothing to install
- No GPU / CUDA is needed

Check the runtime dependencies of the prebuilt core with:

```bash
ldd src/deros_core/lib/libderos_core.so | grep "not found"
```

## Build

    catkin_make -j$(nproc)
    source devel/setup.bash

## Run

The launch files live in `src/deros/launch/datasets` and
`src/deros/launch/online`.

Offline (default, `run_mode=2`: point clouds come from the bag):

    roslaunch deros kitti_start.launch sequence:=00

Online semantic mode (`run_mode=3`: subscribe to the cloud published by a
semantic inference node, see `launch/online/semantic_inference.launch`):

    roslaunch deros kitti_start.launch run_mode:=3 launch_semantic:=true

`run_mode` values:

| value | meaning |
|---|---|
| 0 | ROS_RUN, front-end runs at `ratehz` and does not wait for the back-end |
| 1 | EVAL_ONLY, no ROS output, evaluation only |
| 2 | EVAL_AND_ROS (default), front-end waits for the back-end every frame |
| 3 | ONLINE_SEMANTIC, subscribe to a semantic cloud topic instead of reading a bag |

Every mode writes the odometry and mapping trajectories to `save/evo_log`,
saves the global map to `save/map`, and runs `eval_cmd` when the run ends.

## Notes

- `src/deros/save/evo_log` and `src/deros/save/map` are result directories;
  their contents are not tracked.
- The registration parameters `icp_search_scale` / `icp_kernel_scale` are ROS
  private parameters. They live on the parameter server across node restarts,
  so a value set by one launch file (for example the wider setting used on
  SubT) is inherited by the next run unless the new launch sets it again or the
  ROS master is restarted. If a dataset starts drifting unexpectedly, restart
  `roscore` first.
