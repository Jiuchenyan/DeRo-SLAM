#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <Eigen/Core>
#include <sophus/se3.hpp>

#include "deros_core/labels.hpp"

namespace ros {
class NodeHandle;
}

namespace dero_slam {

/// One frame handed by the odometry thread to the mapping thread.
struct scan_data{
    int cloud_id;
    V3d_i frame;
    std::vector<float> intensity;
    Sophus::SE3d pose;
    Sophus::SE3d model_deviation;
    Eigen::MatrixXi ground_bev_grid;
    Eigen::MatrixXi noground_bev_grid;
    bool deviation_flag;
    double deviation_score;
    double deviation_score_g;    // ground part of the deviation score
    double deviation_score_b;    // building part of the deviation score
    double ground_ratio;         // fraction of ground points in the frame
};

/// Degradation cues handed back by the mapping thread to the odometry thread.
struct mapping_data{
    int gnd_line_count;
    int gnd_its_count;
    std::vector<int> building_orient_bins;
    double building_rho2;
    double building_line_len;
};

// Channels between the odometry thread and the mapping thread.
extern std::queue<scan_data> scan_queue;
extern std::mutex scan_mutex;

extern mapping_data mapping_message;
extern std::mutex mapping_mutex;

extern std::mutex sync_mutex;
extern std::condition_variable sync_cv;
extern bool frame_processed;
extern bool break_sign;

/// How the node is driven, selected by the `run_mode` parameter.
namespace run_mode {
constexpr int kRosRun = 0;      // publish everything, front end paced by ratehz
constexpr int kEvalOnly = 1;    // no ROS output, only the trajectories and the evaluation
constexpr int kEvalAndRos = 2;  // ROS output, front end waits for the mapping thread
constexpr int kOnline = 3;      // point clouds come from a semantic topic instead of the bag
}   // namespace run_mode

/// Name of a run mode, for logging.
const char* RunModeName(int mode);

/// Reads the `run_mode` parameter. The legacy `debug_mode` name is still
/// accepted, so old launch files keep working.
int ReadRunMode(const ros::NodeHandle &pnh);

}   // namespace dero_slam
