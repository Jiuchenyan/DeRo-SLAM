#pragma once

/// @file base_types.hpp
/// Small value types shared by the odometry and the mapping interface.

#include <vector>

#include <Eigen/Core>

#include "deros_core/types.hpp"

#ifndef SOPHUS_USE_BASIC_LOGGING
#define SOPHUS_USE_BASIC_LOGGING 1
#endif
#include <sophus/se3.hpp>

namespace deros_api {

/// Degradation cues handed from the mapping thread to the odometry thread.
struct DeviationMess {
    int gnd_line_count = 0;
    int gnd_its_count = 0;
    bool is_deviation = false;
    std::vector<int> building_orient_bins;
    double building_rho2 = 0.0;
    double building_line_len = 0.0;
    double ground_ratio = 0.0;
    double score_g = 0.0;
    double score_b = 0.0;
    double score_total = 0.0;
};

/// A detected loop closure.
struct LoopResult {
    bool is_valid = false;
    int curr_idx = -1;    // cloud_idx of the current frame
    int match_idx = -1;   // cloud_idx of the matched frame
    Sophus::SE3d relative_pose;
    double rot_noise_scale = 1.0;
    double trans_noise_scale = 1.0;
};

/// One ground-map voxel, as needed by the visualisation of normals.
struct NormalSample {
    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    Eigen::Vector3d normal = Eigen::Vector3d::Zero();
    bool normal_computed = false;
};

struct GroundNormals {
    double voxel_size = 1.0;
    std::vector<NormalSample> samples;
};

}   // namespace deros_api
