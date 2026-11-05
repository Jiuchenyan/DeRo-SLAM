#pragma once

#include <chrono>
#include <fstream>
#include <string>

// ROS
#include "geometry_msgs/PoseStamped.h"
#include "geometry_msgs/TransformStamped.h"
#include "nav_msgs/Odometry.h"
#include "nav_msgs/Path.h"
#include "ros/node_handle.h"
#include "ros/ros.h"
#include "sensor_msgs/PointCloud2.h"
#include "tf2_ros/transform_broadcaster.h"
#include "visualization_msgs/Marker.h"

#include "dataprocess.hpp"
#include "NodeShared.hpp"
#include "online_cloud.hpp"

#include "deros_core/deros_core.hpp"

namespace dero_slam{

/// ROS wrapper around deros_api::SlamOdometry: pulls the scans out of the bag,
/// runs the odometry and publishes the result.
class Odometry{
public:
    Odometry(const ros::NodeHandle& nh,const ros::NodeHandle& pnh);
    ~Odometry();
    void RegisterFrame();
    void SetVizPose(const Sophus::SE3d& pose);
private:
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    std::string data_set_;
    std::string bag_path_;
    std::string lidar_topic_;

    std::string child_frame_{"base_link"};
    std::string odom_frame_{"odom"};
    std::string result_path_{"./"};
    std::ofstream fout_odom_;

    double ratehz_;
    int run_mode_;
    bool using_intensity_;
    deros_api::SlamOdometryConfig slam_odom_cfg_;
    deros_api::SlamOdometry slam_odometry_;

    ros::Publisher allcloud_frame_publisher_;
    ros::Publisher ground_frame_publisher_;
    ros::Publisher noground_frame_publisher_;
    ros::Publisher allcloud_map_publisher_;
    ros::Publisher ground_map_publisher_;
    ros::Publisher noground_map_publisher_;

    ros::Publisher odom_publisher_;
    ros::Publisher traj_publisher_;

    ros::Publisher local_ground_normals_pub_;
    ros::Publisher frame_id_marker_pub_;

    tf2_ros::TransformBroadcaster tf_broadcaster_;
    nav_msgs::Path path_msg_;

    int cloud_idx_;
    ros::Time cloud_time_;
    int start_idx_{0};                   // initial cloud_start_idx, the same skip is applied online

    //!------------------ online semantic clouds (run_mode == kOnline) ---------
    bool online_mode_{false};
    std::string semantic_cloud_topic_;   // topic published by the semantic node
    int online_queue_size_{30};          // buffered frames, the oldest are dropped
    double online_idle_timeout_{5.0};    // end of stream when no cloud arrives for this long (0 = wait forever)
    int online_frame_cnt_{0};
    OnlineCloudBuffer online_cloud_buffer_;
    ros::Subscriber semantic_cloud_sub_;
    std::chrono::steady_clock::time_point online_t_start_{};
    bool online_started_{false};
    double online_wait_sum_ms_{0.0};   // accumulated time spent waiting for clouds
    std::chrono::steady_clock::time_point last_online_msg_t_{};
    double online_wait_ms_{0.0};
    double online_parse_ms_{0.0};

    //!------------------ helpers ----------------------------------------------
    /// Blocks until a scan is available.
    /// @return 1 = frame fetched, 0 = empty frame skipped,
    ///         -1 = shutting down, -2 = end of the stream (idle timeout).
    int FetchOnlineFrame(V3d_i &frame_raw, std::vector<float> &intensity_raw);
    void OnlineCloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg);

private:
    void PublishTF(const Sophus::SE3d& pose, const std::string& frame_id, const std::string& child_frame_id, const ros::Time& stamp);
    void PublishOdom(const Sophus::SE3d& pose, const std::string& frame_id, const std::string& child_frame_id, const ros::Time& stamp,const bool& is_pub_traj);
    void SaveOdomRes(std::ofstream& fout_odom, const std::string& result_path,const Sophus::SE3d& pose);
    void PublishClds(const deros_api::CloudRes& res2pub, const Sophus::SE3d& corr);
    void PublishNormals(const deros_api::GroundNormals& normals,const ros::Publisher& pub, const ros::Time& stamp,const std::string& frame_id,const Sophus::SE3d& corr);
    void PublishFrameIdMarker(const Sophus::SE3d& pose, const std::string& frame_id, const ros::Time& stamp, int cloud_id);

    std::mutex viz_mutex_;
    Sophus::SE3d viz_pose_;
    bool viz_ready_ = false;
};

}   // namespace dero_slam
