#pragma once

/// @file pose_graph.hpp
/// Public interface of the pose-graph back-end (incremental smoother).

#include <memory>
#include <vector>

#ifndef SOPHUS_USE_BASIC_LOGGING
#define SOPHUS_USE_BASIC_LOGGING 1
#endif
#include <sophus/se3.hpp>

namespace deros_api {

/// Incremental pose graph. Wraps the optimiser that lives in the library.
class PoseGraphOptimizer {
public:
    PoseGraphOptimizer();
    ~PoseGraphOptimizer();

    PoseGraphOptimizer(PoseGraphOptimizer &&) noexcept;
    PoseGraphOptimizer &operator=(PoseGraphOptimizer &&) noexcept;

    PoseGraphOptimizer(const PoseGraphOptimizer &) = delete;
    PoseGraphOptimizer &operator=(const PoseGraphOptimizer &) = delete;

    void AddPriorFactor(int frame_idx, const Sophus::SE3d &pose);

    void AddOdometryEdge(int prev_idx, int curr_idx,
                         const Sophus::SE3d &relative_pose,
                         double rot_sigma, double trans_sigma);

    void AddLoopClosureEdge(int match_idx, int curr_idx,
                            const Sophus::SE3d &relative_pose,
                            double rot_sigma, double trans_sigma);

    void SetInitialPose(int frame_idx, const Sophus::SE3d &pose);

    void Optimize();

    Sophus::SE3d GetOptimizedPose(int frame_idx);

    std::vector<Sophus::SE3d> GetAllOptimizedPoses(int max_idx);

    void Reset();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}   // namespace deros_api
