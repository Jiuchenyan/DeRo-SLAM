# Build & Run

This repository is the **release package** of DeRo-SLAM. The algorithm sources
are not included: they ship precompiled as
`src/deros_core/lib/libderos_core.so` together with the public headers in
`src/deros_core/include`. `src/deros` is the ROS application layer and can be
read and modified freely.

## Requirements

- Ubuntu 20.04 + ROS Noetic
- OpenCV 4, TBB, glog, OpenBLAS, Suitesparse and Boost (the shared libraries
  `libderos_core.so` links against, plus the Boost components GTSAM's CMake
  config requires):

```bash
sudo apt install libgoogle-glog-dev libsuitesparse-dev libopenblas-dev \
                 libtbb-dev libopencv-dev libboost-all-dev
```

- GTSAM **4.2.0 built with TBB** (`libgtsam.so.4`) — see
  [GTSAM](#gtsam-required) below
- Eigen and Sophus headers are vendored under
  `src/deros_core/include/third_party` — nothing to install
- No GPU / CUDA is needed

Check the runtime dependencies of the prebuilt core with:

```bash
ldd src/deros_core/lib/libderos_core.so | grep "not found"
```

### GTSAM (required)

`libderos_core.so` is linked against `libgtsam.so.4` and exports symbols that
depend on its exact ABI, including the TBB allocator
(`std::vector<... tbb::tbb_allocator<...>>`). GTSAM built **without** TBB, or a
different version such as 4.0.3 / 4.1.1, fails at link time with undefined
references to `gtsam::*`.

```bash
git clone -b 4.2.0 --depth 1 https://github.com/borglab/gtsam.git
cmake -S gtsam -B gtsam/build \
      -DGTSAM_USE_TBB=ON \
      -DGTSAM_BUILD_EXAMPLES_AND_TESTS=OFF \
      -DGTSAM_BUILD_UNSTABLE=OFF \
      -DCMAKE_INSTALL_PREFIX=/usr/local
cmake --build gtsam/build -j$(nproc)
sudo cmake --install gtsam/build
sudo ldconfig
```

`libtbb-dev` must already be installed **before** the configure step: if TBB is
missing, GTSAM quietly builds without it and the resulting library still fails
to link against `libderos_core.so`. Verify the result before going further:

```bash
nm -D -C /usr/local/lib/libgtsam.so.4 | grep -c tbb_allocator   # must be > 0
ldconfig -p | grep libgtsam.so.4                                 # must list /usr/local/lib
```

If GTSAM is installed somewhere else, tell CMake where to look:

```bash
catkin_make -DGTSAM_DIR=/path/to/gtsam/lib/cmake/GTSAM
# or, if there is no CMake config file:
catkin_make -DGTSAM_LIBRARY=/path/to/libgtsam.so.4
```

A distribution GTSAM under `/usr` (4.0.3 / 4.1.1) does not help and is shadowed
by the `/usr/local` copy. `cmake` prints which one it resolved:

```
-- GTSAM: /usr/local/lib/cmake/GTSAM (CMake config)
```

If GTSAM is missing, `cmake` now stops with this message instead of failing
later in the linker.

## Build

    catkin_make -j$(nproc)
    source devel/setup.bash

## Run

The launch files live in `src/deros/launch/datasets` and
`src/deros/launch/online`. Each one expects its dataset under `~/data/<dataset>`
by default; override with `database_path:=/your/path`.

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
