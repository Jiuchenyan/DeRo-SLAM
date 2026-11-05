# DeRo-SLAM

**DeRo-SLAM** is a LiDAR SLAM system that couples semantic segmentation with a
degradation-aware front-end: the front-end monitors the ground and building
cues of every scan, and switches the registration strategy when the scene is
degraded (open spaces, tunnels, corridors) instead of trusting a single ICP
setting everywhere.

> **Release package.** This repository ships the algorithm core as a precompiled
> library (`src/deros_core/lib/libderos_core.so`) together with its public
> headers. The ROS application layer (`src/deros`) is plain source and can be
> read and modified freely. The full algorithm source will be published after
> the paper is accepted.

## Repository layout

```
.
├── README.md
├── BUILD.md                     build and run instructions
└── src/
    ├── deros_core/
    │   ├── include/deros_core/  public API headers (data structures only)
    │   ├── include/third_party/ vendored Eigen and Sophus headers
    │   └── lib/                 libderos_core.so
    └── deros/                   ROS application layer
        ├── launch/datasets/     one launch file per dataset
        ├── launch/online/       online semantic input + inference nodes
        ├── scripts/             evaluation and semantic inference scripts
        ├── src/                 OdometryNode / MappingNode
        └── save/                where trajectories and maps are written
```

## Quick start

```bash
catkin_make -j$(nproc)
source devel/setup.bash

# offline, point clouds and labels come from the bag
roslaunch deros kitti_start.launch sequence:=00
```

Results are written to `src/deros/save/evo_log/` (trajectories) and
`src/deros/save/map/` (global map). Every run prints the ATE metrics at the end.

See **[BUILD.md](BUILD.md)** for requirements, the full list of launch
arguments and the `run_mode` description.

## Supported datasets

| dataset | launch file |
|---|---|
| KITTI | `kitti_start.launch sequence:=00` |
| KITTI-360 | `kitti360_start.launch sequence:=00` |
| MulRan | `mulran_start.launch sequence:=...` |
| SubT-MRS | `subt_start.launch` |
| Goose-3D (Geode) | `geode_start.launch` |
| TrackLoop (custom) | `trackloop_start.launch` |
| MID360 | `mid360.launch` |

## Online semantic mode

`run_mode:=3` replaces the bag reader with a subscriber to a semantic point
cloud topic. A Cylinder3D / RandLA-Net / SegNet4D inference node can be started
by the launch file itself:

```bash
roslaunch deros kitti_start.launch run_mode:=3 launch_semantic:=true
```

The inference nodes need their own Python environments and model weights; see
`src/deros/scripts/semantic/README.md`.

## Citation

```bibtex
@article{dero_slam,
  title   = {DeRo-SLAM},
  author  = {TBD},
  journal = {TBD},
  year    = {TBD}
}
```

## License

To be determined. Until a license is added, all rights are reserved.
