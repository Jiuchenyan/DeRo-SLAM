#pragma once

/// @file mapping.hpp
/// Public interface of the DeRo-SLAM mapping back-end (descriptors, loop
/// detection, pose graph).
///
/// As with the odometry header, all the actual work happens inside
/// libderos_core.so.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <opencv2/core.hpp>

#ifndef SOPHUS_USE_BASIC_LOGGING
#define SOPHUS_USE_BASIC_LOGGING 1
#endif
#include <sophus/se3.hpp>

#include "deros_core/base_types.hpp"

namespace deros_api {

/// Parameters of the back-end.
struct SlamMappingConfig {
    double max_range = 100.0;
    double min_range = 5.0;
    double bev_grid_size = 0.2;
    int close_kernel_size = 3;
    int loop_num_exclude = 30;
    int loop_num_search = 5;
    std::string loop_backend = "semantic";   // semantic | scancontext
    double out_dists_th = 0.05;
    std::string save_path = "./";

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 0.1;
    double max_expected_sigma = 2.0;
    int instance_threshold = 3;

    int nFeatures = 300;
    float scaleFactor = 1.2f;
    int nLevels = 4;
    int iniThFAST = 30;
    int minThFAST = 10;
    int max_points_per_voxel = 20;
    double voxel_size = 1.0;

    int keyframe_interval = 5;
    bool adaptive_weights = true;
    int pgo_frame_interval = 10;
    bool enable_pgo = true;
    double odom_rot_noise = 0.1;
    double odom_trans_noise = 0.3;
    double loop_rot_noise = 0.5;
    double loop_trans_noise = 1.0;
};

/// Everything the back-end produces for one frame.
struct MappingRes {
    LoopResult loop;
    int gnd_line_count = 0;
    int gnd_its_count = 0;
    std::vector<int> building_orient_bins;
    double building_rho2 = 0.0;
    double building_line_len = 0.0;
    cv::Mat ground_line_in_map;
    cv::Mat building_line_in_map;   // building lines + clusters, shown in rviz
};

/// The back-end. One instance handles one sequence.
class SlamMapping {
public:
    explicit SlamMapping(const SlamMappingConfig &config);
    SlamMapping();
    ~SlamMapping();

    SlamMapping(SlamMapping &&) noexcept;
    SlamMapping &operator=(SlamMapping &&) noexcept;

    SlamMapping(const SlamMapping &) = delete;
    SlamMapping &operator=(const SlamMapping &) = delete;

    MappingRes mainProcess(const V3d_i &frame,
                           const Sophus::SE3d &pose,
                           const std::vector<float> &intensity,
                           const Sophus::SE3d &pose_deviation,
                           const Eigen::MatrixXi &gnd_bev,
                           const Eigen::MatrixXi &nognd_bev,
                           int cloud_idx);

    /// Indices and clouds of every frame kept as a keyframe.
    std::pair<std::vector<int>, std::vector<V3d_i>> ReturnAllFrames() const;

    /// Re-anchor the stored descriptors after the poses were optimised.
    void UpdateAllPoseDescriptors(const std::vector<Sophus::SE3d> &optimized_poses);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace deros_api
