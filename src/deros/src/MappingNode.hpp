#pragma once

#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

// ROS
#include "geometry_msgs/PoseStamped.h"
#include "nav_msgs/Path.h"
#include "ros/node_handle.h"
#include "ros/ros.h"
#include "sensor_msgs/Image.h"
#include "sensor_msgs/PointCloud2.h"
#include "visualization_msgs/Marker.h"
#include "visualization_msgs/MarkerArray.h"

#include "dataprocess.hpp"
#include "NodeShared.hpp"

#include "deros_core/deros_core.hpp"

namespace dero_slam{

/// Everything the debug image shows about the deviation check of one frame.
struct DebugImageInfo{
    bool dev_flag = false;
    double dev_score = 0.0;        // total score
    int gnd_line = 0;
    int gnd_its = 0;
    double ground_ratio = 0.0;     // fraction of ground points
    double ground_score = 0.0;     // ground part of the score
    double building_rho2 = 0.0;
    double building_len = 0.0;
    double building_score = 0.0;   // building part of the score
};

class Odometry;

/// ROS wrapper around deros_api::SlamMapping: consumes the odometry frames, runs
/// the descriptor/loop-closure backend and publishes the map.
class Mapping{
public:
    void ScanMapping();
    Mapping(const ros::NodeHandle& nh,const ros::NodeHandle& pnh);
    ~Mapping();
    void SetOdometry(Odometry* odom) { odom_ = odom; }
    void SaveGlobalMap(const std::string& ply_path,
                       const std::string& label_path,
                       double voxel_size = 0.3,
                       int step = 5);

private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Publisher traj_pgo_publisher_;
    ros::Publisher traj_grad_publisher_;
    ros::Publisher loop_edge_publisher_;

    double traj_line_width_ = 2.0;
    ros::Publisher global_map_pub_;
    ros::Publisher ground_line_pub_;
    ros::Publisher building_pub_;
    ros::Publisher debug_info_pub_;
    deros_api::SlamMapping slam_mapping_;
    deros_api::SlamMappingConfig slam_mapping_cfg_;
    deros_api::PoseGraphOptimizer pose_graph_;
    Odometry* odom_ = nullptr;

    int last_cloud_idx_ = -1;
    Sophus::SE3d last_pose_;
    Sophus::SE3d latest_opt_pose_;
    Sophus::SE3d cul_ref_odom_ = Sophus::SE3d();

    int frame_counter_ = 0;
    int loop_skip_counter_ = 0;
    std::vector<Sophus::SE3d> poses_pgo_;
    std::vector<std::pair<int, int>>  loop_pair_vec_;

    std::string child_frame_{"base_link"};
    std::string odom_frame_{"odom"};
    std::string result_path_{"./"};
    std::ofstream fout_mapping_;

    std::string eval_cmd_{""};
    int run_mode_;
    bool save_global_map_;
    bool enable_eval_{true};          // every run mode evaluates at the end
    void PublishTrajectory();
    void visualizeLoopClosure();
    void SaveOptimizedTrajectory();
    void PublishGlobalMap();
    void PublishImage(const ros::Publisher& pub, const cv::Mat& image, const ros::Time& stamp, const std::string& frame_id);
    void EvoEval() const;
    void WritePLY(const std::string& file_path,
                  const std::vector<Eigen::Vector3d>& points,
                  const std::vector<int>& labels,
                  const std::unordered_map<int, Eigen::Vector3i>& color_map);
    cv::Mat CreateDebugImage(const DebugImageInfo &info);
};

}   // namespace dero_slam
