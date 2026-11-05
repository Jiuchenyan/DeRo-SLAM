#pragma once

/// @file odometry.hpp
/// Public interface of the DeRo-SLAM front-end (odometry).
///
/// This header is the *only* thing an application needs in order to run the
/// front-end: every implementation detail lives inside libderos_core.so, behind
/// the pimpl pointer. Nothing here exposes the registration, descriptor or
/// optimisation internals.

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

// Sophus is header only; use the dependency free logging path so that a
// consumer does not have to have fmt installed.
#ifndef SOPHUS_USE_BASIC_LOGGING
#define SOPHUS_USE_BASIC_LOGGING 1
#endif
#include <sophus/se3.hpp>

#include "deros_core/base_types.hpp"

namespace deros_api {

/// Parameters of the front-end. The defaults are the published ones.
struct SlamOdometryConfig {
    // map params
    double voxel_size = 1.0;
    double max_range = 100.0;
    double min_range = 5.0;
    int max_points_per_voxel = 20;

    // th parms
    double min_motion_th = 0.1;
    double initial_threshold = 2.0;

    bool remove_foreground = false;

    // ICP termination
    int icp_max_iterations = 500;
    double icp_tolerance = 1e-4;

    // Registration support, both expressed relative to the adaptive threshold
    // sigma.
    //   correspondence gate = icp_search_scale * sigma   (3.0)
    //   robust kernel width = sigma / icp_kernel_scale   (3.0)
    double icp_search_scale = 3.0;
    double icp_kernel_scale = 3.0;
};

/// Everything the front-end produces for one frame.
struct CloudRes {
    V3d_i process_frame;
    std::vector<float> process_intensity;
    V4d All_frame;
    V4d ground_frame;
    V4d no_ground_frame;
    Eigen::MatrixXi noground_bev_grid;
    Eigen::MatrixXi ground_bev_grid;
    Sophus::SE3d model_deviation;
    bool deviation_flag = false;
    double deviation_score = 0.0;
    // Split of the deviation score, filled for the debug image.
    double deviation_score_g = 0.0;   // ground part
    double deviation_score_b = 0.0;   // building part
    double ground_ratio = 0.0;        // fraction of ground points in the frame
};

/// The front-end. One instance handles one sequence.
///
/// Non copyable: the implementation owns large buffers and is meant to be
/// constructed once per run (it is movable, which is what the ROS layer uses
/// when it re-configures the solver from the parameter server).
class SlamOdometry {
public:
    explicit SlamOdometry(const SlamOdometryConfig &config);
    SlamOdometry();
    ~SlamOdometry();

    SlamOdometry(SlamOdometry &&) noexcept;
    SlamOdometry &operator=(SlamOdometry &&) noexcept;

    SlamOdometry(const SlamOdometry &) = delete;
    SlamOdometry &operator=(const SlamOdometry &) = delete;

    /// Run the whole front-end on one frame: preprocess, align against the
    /// local map, update the map and return the result.
    CloudRes mainProcess(const V3d &frame,
                         const std::vector<int> &frame_label,
                         const std::vector<float> &intensity,
                         const std::string &data_set);

    /// Degradation cues coming back from the mapping thread.
    void UpdateDeviationMess(const DeviationMess &new_mess);

    /// Voxelised (map, register) pair of a frame, for callers that want to
    /// prepare clouds on their own.
    std::pair<V3d_i, V3d_i> VoxelizeSemantic(const V3d_i &frame,
                                            const std::string &data_set) const;

    /// Current local maps, in the local frame.
    V4d LocalMap() const;
    V4d GroundMap() const;
    V4d NoGroundMap() const;

    /// Ground map centroids and normals, for visualisation only.
    GroundNormals GroundNormalsMap() const;

    /// Poses of all frames processed so far.
    std::vector<Sophus::SE3d> poses() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace deros_api
