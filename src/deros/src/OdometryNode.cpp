#include "OdometryNode.hpp"

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <utility>

#include "dataprint.hpp"

namespace dero_slam{

namespace {

// The trajectory files are opened with trunc/app, so a missing directory would
// silently swallow them and the evaluation would fail on a missing file.
void EnsureParentDir(const std::string &path) {
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
}

}   // namespace

Odometry::Odometry(const ros::NodeHandle& nh,const ros::NodeHandle& pnh):nh_(nh),pnh_(pnh){
    pnh_.param<std::string>("data_set",data_set_,data_set_);
    pnh_.param<std::string>("bag_path",bag_path_,bag_path_);
    pnh_.param<std::string>("lidar_topic",lidar_topic_,lidar_topic_);
    
    pnh_.param<std::string>("odometry_result_path",result_path_,result_path_);
    EnsureParentDir(result_path_);
    pnh_.param<std::string>("child_frame", child_frame_, child_frame_); // child frame name, i.e., velodyne frame
    pnh_.param<std::string>("odom_frame", odom_frame_, odom_frame_);    // odom frame name
    pnh_.param<double>("ratehz", ratehz_, 10.0);    // odom frame name
    run_mode_ = ReadRunMode(pnh_);
    pnh_.param<int>("cloud_start_idx", cloud_idx_, 0);
    start_idx_ = cloud_idx_;
    pnh_.param<bool>("using_intensity", using_intensity_, false);

    //!------------------ online semantic clouds -----------------------------
    pnh_.param<std::string>("semantic_cloud_topic", semantic_cloud_topic_, semantic_cloud_topic_);
    pnh_.param<int>("online_queue_size", online_queue_size_, online_queue_size_);
    pnh_.param<double>("online_idle_timeout", online_idle_timeout_, online_idle_timeout_);
    online_mode_ = (run_mode_ == run_mode::kOnline);


    pnh_.param<double>("max_range",slam_odom_cfg_.max_range,slam_odom_cfg_.max_range);
    pnh_.param<double>("min_range",slam_odom_cfg_.min_range,slam_odom_cfg_.min_range);
    pnh_.param<double>("voxel_size",slam_odom_cfg_.voxel_size,slam_odom_cfg_.voxel_size);
    pnh_.param<int>("max_points_per_voxel", slam_odom_cfg_.max_points_per_voxel, slam_odom_cfg_.max_points_per_voxel);
    pnh_.param<bool>("remove_foreground", slam_odom_cfg_.remove_foreground, slam_odom_cfg_.remove_foreground);
    pnh_.param<int>("icp_max_iterations", slam_odom_cfg_.icp_max_iterations, slam_odom_cfg_.icp_max_iterations);
    pnh_.param<double>("icp_tolerance", slam_odom_cfg_.icp_tolerance, slam_odom_cfg_.icp_tolerance);
    pnh_.param<double>("icp_search_scale", slam_odom_cfg_.icp_search_scale, slam_odom_cfg_.icp_search_scale);
    pnh_.param<double>("icp_kernel_scale", slam_odom_cfg_.icp_kernel_scale, slam_odom_cfg_.icp_kernel_scale);

    slam_odometry_ = deros_api::SlamOdometry(slam_odom_cfg_);

    fout_odom_.open(result_path_,std::ofstream::trunc);
    fout_odom_.close(); // clear the file

    if(run_mode_ == run_mode::kEvalOnly) return;
    //!------------------ ROS -------------------------

    allcloud_frame_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("allcloud_frame", 1); // current frame publisher
    allcloud_map_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("allcloud_map", 1); // current frame publisher

    ground_frame_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("ground_frame", 1); // current frame publisher
    noground_frame_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("noground_frame", 1); // current frame publisher

    ground_map_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("ground_map", 1); // current frame publisher
    noground_map_publisher_ = pnh_.advertise<sensor_msgs::PointCloud2>("noground_map", 1); // current frame publisher

    local_ground_normals_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>("local_ground_normals", 1, true);

    frame_id_marker_pub_ = pnh_.advertise<visualization_msgs::Marker>("frame_id_markers", 100);

    odom_publisher_ = pnh_.advertise<nav_msgs::Odometry>("odometry", 100);   // odometry publisher
    traj_publisher_ = pnh_.advertise<nav_msgs::Path>("trajectory", 100);

    if(online_mode_){
        if(semantic_cloud_topic_.empty()){
            mprtln(RED, "[online error] run_mode=3 needs a non-empty semantic_cloud_topic");
            std::exit(EXIT_FAILURE);
        }
        online_cloud_buffer_.SetMaxSize(static_cast<size_t>(std::max(1, online_queue_size_)));
        semantic_cloud_sub_ = nh_.subscribe<sensor_msgs::PointCloud2>(
            semantic_cloud_topic_, static_cast<uint32_t>(std::max(1, online_queue_size_)),
            &Odometry::OnlineCloudCallback, this, ros::TransportHints().tcpNoDelay());
        mprtln(GREEN, "[online] subscribing %s | queue %d | idle timeout %.1f s",
               semantic_cloud_topic_.c_str(), online_queue_size_, online_idle_timeout_);
    }
}

Odometry::~Odometry(){}

void Odometry::RegisterFrame(){
    ros::Rate loop(ratehz_);
    while(ros::ok()){
        std::vector<float> intensity_raw;
        V3d_i frame_raw;
        auto ts = std::chrono::steady_clock::now();
        if(online_mode_){
            //!------------------ online: fetch from the semantic topic ----------
            const int status = FetchOnlineFrame(frame_raw, intensity_raw);
            if(status < 0){
                if(status == -2){
                    const double sec = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - online_t_start_).count();
                    mprtln(GREEN, "[online] data stream finished: %d frames in %.1f s = %.2f fps"
                            " | mean wait %.1f ms | dropped %zu",
                           cloud_idx_, sec, cloud_idx_ / std::max(1e-6, sec),
                           online_wait_sum_ms_ / std::max(1, cloud_idx_), online_cloud_buffer_.Dropped());
                }
                break_sign = true;
                break;
            }
            if(status == 0) continue;
            ++cloud_idx_;
        } else {
            frame_raw = LoadData(bag_path_,lidar_topic_,intensity_raw,cloud_idx_,data_set_);
            cloud_time_ = ros::Time::now();
        }

        if (frame_raw.first.empty()) {
            mprtln(YELLOW,"No more data, shutting down");
            break_sign = true;
            break; 
        }
        
        deros_api::DeviationMess deviation_mess;
        mapping_mutex.lock();
        deviation_mess.gnd_line_count = mapping_message.gnd_line_count;
        deviation_mess.gnd_its_count = mapping_message.gnd_its_count;
        deviation_mess.building_orient_bins = mapping_message.building_orient_bins;
        deviation_mess.building_rho2 = mapping_message.building_rho2;
        deviation_mess.building_line_len = mapping_message.building_line_len;
        mapping_mutex.unlock();
        slam_odometry_.UpdateDeviationMess(deviation_mess);

        auto t1 = std::chrono::steady_clock::now();
        const auto& res2pub = slam_odometry_.mainProcess(frame_raw.first,frame_raw.second,intensity_raw,data_set_);
        auto t2 = std::chrono::steady_clock::now();

        const auto pose = slam_odometry_.poses().back(); 


        scan_data data;
        data.cloud_id = cloud_idx_;
        data.intensity = res2pub.process_intensity;
        data.frame = res2pub.process_frame;
        data.pose = pose;
        data.model_deviation = res2pub.model_deviation;
        data.ground_bev_grid = res2pub.ground_bev_grid;
        data.noground_bev_grid = res2pub.noground_bev_grid;
        data.deviation_flag = res2pub.deviation_flag;
        data.deviation_score = res2pub.deviation_score;
        data.deviation_score_g = res2pub.deviation_score_g;
        data.deviation_score_b = res2pub.deviation_score_b;
        data.ground_ratio = res2pub.ground_ratio;

        scan_mutex.lock();
        scan_queue.push(std::move(data)); 
        scan_mutex.unlock();


        if (run_mode_ != run_mode::kRosRun) {
            std::unique_lock<std::mutex> lock(sync_mutex);
            sync_cv.wait(lock, [] { return frame_processed; });
            frame_processed = false;
        }
        const auto t_sync = std::chrono::steady_clock::now();

        SaveOdomRes(fout_odom_,result_path_,pose);
        static double avg_t=0.0;
        static int frame_cnt = 0;
        if(run_mode_==run_mode::kEvalOnly){
            double delta_t_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
            frame_cnt++;
            avg_t = avg_t + (delta_t_ms - avg_t) / frame_cnt;
            mprtln(MAGENTA, "[Scan] cloud_idx: %d | main_time:%.3f ms | avg_time:%.3f ms", cloud_idx_, delta_t_ms, avg_t);
            ros::spinOnce();
            continue;
        } 
        //!------------------ ROS -------------------------
        // Publish
        Sophus::SE3d viz_pose = pose;
        {
            std::lock_guard<std::mutex> lock(viz_mutex_);
            if (viz_ready_) viz_pose = viz_pose_;
        }
        const Sophus::SE3d viz_corr = viz_pose * pose.inverse();

        PublishTF(viz_pose,odom_frame_,child_frame_,cloud_time_);
        PublishOdom(pose,odom_frame_,child_frame_,cloud_time_,true);
        PublishClds(res2pub, viz_corr);
        PublishNormals(slam_odometry_.GroundNormalsMap(),local_ground_normals_pub_,cloud_time_,odom_frame_,viz_corr);
        PublishFrameIdMarker(viz_pose, odom_frame_, cloud_time_, cloud_idx_);
        if(run_mode_==run_mode::kEvalAndRos || online_mode_){
            const auto te = std::chrono::steady_clock::now();
            const auto ms = [](auto d){ return std::chrono::duration<double, std::milli>(d).count(); };
            if(online_mode_){
                mprtln(MAGENTA, "[online] idx:%d | wait:%.1f | parse:%.1f | main:%.1f | sync:%.1f | all:%.1f ms | queue:%zu",
                       cloud_idx_, online_wait_ms_, online_parse_ms_, ms(t2 - t1),
                       ms(t_sync - t2), ms(te - ts), online_cloud_buffer_.Size());
                if(cloud_idx_ % 100 == 0){
                    const double sec = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - online_t_start_).count();
                    mprtln(CYAN, "[online] progress: %d frames processed | %.2f fps | received %d | dropped %zu | queue %zu",
                           cloud_idx_, cloud_idx_ / std::max(1e-6, sec), online_frame_cnt_,
                           online_cloud_buffer_.Dropped(), online_cloud_buffer_.Size());
                }
            } else {
                mprtln(MAGENTA,"[Scan] cloud_idx: %d | main_time:%.3f ms",cloud_idx_,(float)(t2-t1).count()*1e-6);
            }
            ros::spinOnce();
            continue;
        } 

        auto te = std::chrono::steady_clock::now();
        mprtln(CYAN,"[Scan] cloud_idx: %d | main_time: %.3f ms | all_time: %.3f ms", cloud_idx_, (float)(t2-t1).count()*1e-6,(float)(te-ts).count()*1e-6);
        if(!online_mode_) loop.sleep();
        ros::spinOnce();
    }
}


void Odometry::SetVizPose(const Sophus::SE3d& pose){
    std::lock_guard<std::mutex> lock(viz_mutex_);
    viz_pose_ = pose;
    viz_ready_ = true;
}

void Odometry::PublishTF(const Sophus::SE3d& pose, const std::string& frame_id, const std::string& child_frame_id, const ros::Time& stamp){

    // Convert from Eigen to ROS types
    const Eigen::Vector3d& t_current = pose.translation(); 
    const Eigen::Quaterniond& q_current = pose.unit_quaternion();

    geometry_msgs::TransformStamped transform_msg;
    transform_msg.header.stamp = stamp;
    transform_msg.header.frame_id = frame_id;
    transform_msg.child_frame_id = child_frame_id;
    transform_msg.transform.rotation.x = q_current.x();
    transform_msg.transform.rotation.y = q_current.y();
    transform_msg.transform.rotation.z = q_current.z();
    transform_msg.transform.rotation.w = q_current.w();
    transform_msg.transform.translation.x = t_current.x();
    transform_msg.transform.translation.y = t_current.y();
    transform_msg.transform.translation.z = t_current.z();
    tf_broadcaster_.sendTransform(transform_msg);
}

void Odometry::PublishOdom(const Sophus::SE3d& pose, const std::string& frame_id, const std::string& child_frame_id, const ros::Time& stamp,const bool& is_pub_traj){
    nav_msgs::Odometry odom_msg;
    if (odom_publisher_.getNumSubscribers() != 0 || (is_pub_traj && traj_publisher_.getNumSubscribers() != 0)){
        const Eigen::Vector3d& t_current = pose.translation(); 
        const Eigen::Quaterniond& q_current = pose.unit_quaternion();

        // Publish odometry msg
        odom_msg.header.stamp = stamp;
        odom_msg.header.frame_id = frame_id;
        odom_msg.child_frame_id = child_frame_id;
        odom_msg.pose.pose.orientation.x = q_current.x();
        odom_msg.pose.pose.orientation.y = q_current.y();
        odom_msg.pose.pose.orientation.z = q_current.z();
        odom_msg.pose.pose.orientation.w = q_current.w();
        odom_msg.pose.pose.position.x = t_current.x();
        odom_msg.pose.pose.position.y = t_current.y();
        odom_msg.pose.pose.position.z = t_current.z();
        odom_publisher_.publish(odom_msg);
    }
    if (is_pub_traj && traj_publisher_.getNumSubscribers() != 0){
        // publish trajectory msg
        geometry_msgs::PoseStamped pose_msg;
        pose_msg.pose = odom_msg.pose.pose;
        pose_msg.header = odom_msg.header;

        path_msg_.header.stamp = stamp;
        path_msg_.header.frame_id = frame_id;
        path_msg_.poses.push_back(pose_msg);
        traj_publisher_.publish(path_msg_);
    }
}


void Odometry::PublishFrameIdMarker(const Sophus::SE3d& pose, const std::string& frame_id, const ros::Time& stamp, int cloud_id){
    if(frame_id_marker_pub_.getNumSubscribers() == 0) return;
    if(cloud_id%20!=0) return;

    visualization_msgs::Marker marker;
    marker.header.frame_id = frame_id;
    marker.header.stamp = stamp;
    marker.ns = "frame_id_markers";
    marker.id = cloud_id;
    marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
    marker.action = visualization_msgs::Marker::ADD;

    marker.pose.position.x = pose.translation().x();
    marker.pose.position.y = pose.translation().y();
    marker.pose.position.z = pose.translation().z() + 0.5;
    marker.pose.orientation.w = 1.0;

    marker.text = std::to_string(cloud_id);
    marker.scale.z = 3.0;
    marker.color.r = 0.0f;
    marker.color.g = 1.0f;
    marker.color.b = 0.0f;
    marker.color.a = 1.0f;

    marker.lifetime = ros::Duration(0);

    frame_id_marker_pub_.publish(marker);
}


void Odometry::SaveOdomRes(std::ofstream& fout_odom, const std::string& result_path,const Sophus::SE3d& pose){
    if(result_path.empty()) return;

    // Save odometry result, kitti format
    Eigen::Matrix4d poseMat = pose.matrix();
    fout_odom.open(result_path,std::ofstream::app);
    fout_odom.setf(std::ios::fixed, std::ios::floatfield);
    fout_odom.precision(16);
    fout_odom << poseMat(0,0) << " " << poseMat(0,1) << " " << poseMat(0,2) << " " << poseMat(0,3) << " "
            << poseMat(1,0) << " " << poseMat(1,1) << " " << poseMat(1,2) << " " << poseMat(1,3) << " "
            << poseMat(2,0) << " " << poseMat(2,1) << " " << poseMat(2,2) << " " << poseMat(2,3) << std::endl;
    fout_odom.close();
}

void Odometry::PublishClds(const deros_api::CloudRes& res2pub, const Sophus::SE3d& corr){
    if(allcloud_frame_publisher_.getNumSubscribers() !=0)
        allcloud_frame_publisher_.publish(buildXYZRGBLCloud(res2pub.All_frame,"velodyne",cloud_time_));
    if(ground_frame_publisher_.getNumSubscribers() !=0)
        ground_frame_publisher_.publish(buildXYZRGBLCloud(res2pub.ground_frame,"velodyne",cloud_time_));
    if(noground_frame_publisher_.getNumSubscribers() !=0)
        noground_frame_publisher_.publish(buildXYZRGBLCloud(res2pub.no_ground_frame,"velodyne",cloud_time_));

    auto shift_pts = [&corr](std::vector<Eigen::Vector4d> pts) {
        for (auto& p : pts) {
            const Eigen::Vector3d w = corr * Eigen::Vector3d(p.x(), p.y(), p.z());
            p.x() = w.x(); p.y() = w.y(); p.z() = w.z();
        }
        return pts;
    };

    if(allcloud_map_publisher_.getNumSubscribers() !=0) 
        allcloud_map_publisher_.publish(buildXYZRGBLCloud(shift_pts(slam_odometry_.LocalMap()),odom_frame_,cloud_time_));
    if(ground_map_publisher_.getNumSubscribers() !=0) 
        ground_map_publisher_.publish(buildXYZRGBLCloud(shift_pts(slam_odometry_.GroundMap()),odom_frame_,cloud_time_));
    if(noground_map_publisher_.getNumSubscribers() !=0) 
        noground_map_publisher_.publish(buildXYZRGBLCloud(shift_pts(slam_odometry_.NoGroundMap()),odom_frame_,cloud_time_));
}


void Odometry::PublishNormals(const deros_api::GroundNormals& normals,const ros::Publisher& pub, const ros::Time& stamp,const std::string& frame_id,const Sophus::SE3d& corr){
    if (pub.getNumSubscribers() == 0)
        return;

    const double normal_length = normals.voxel_size * 0.8;
    const int sample_num_per_line = 5;
    const int LABEL_GREEN = 102;
    const int LABEL_RED   = 101;
    std::vector<Eigen::Vector4d> normal_points;

    for (const auto& sample : normals.samples) {

        const Eigen::Vector3d& pos    = sample.centroid;
        const Eigen::Vector3d& normal = sample.normal;
        const bool normal_computed = sample.normal_computed;

        const Eigen::Vector3d pos_viz = corr * pos;

        if (!normal_computed) {
            normal_points.emplace_back(pos_viz.x(), pos_viz.y(), pos_viz.z(), LABEL_RED);
        } else {
            Eigen::Vector3d start = pos_viz;
            Eigen::Vector3d end   = pos_viz + corr.rotationMatrix() * (normal * normal_length);
            for (int i = 0; i < sample_num_per_line; ++i) {
                double t = static_cast<double>(i) / (sample_num_per_line - 1);
                Eigen::Vector3d p = start + t * (end - start);
                normal_points.emplace_back(p.x(), p.y(), p.z(), LABEL_GREEN);
            }
        }
    }

    sensor_msgs::PointCloud2 cloud_msg = buildXYZRGBLCloud(normal_points, frame_id, stamp);
    pub.publish(cloud_msg);
}

void Odometry::OnlineCloudCallback(const sensor_msgs::PointCloud2ConstPtr &msg){
    online_cloud_buffer_.Push(msg);
}

int Odometry::FetchOnlineFrame(V3d_i &frame_raw, std::vector<float> &intensity_raw){
    frame_raw.first.clear();
    frame_raw.second.clear();
    intensity_raw.clear();
    online_wait_ms_ = 0.0;
    online_parse_ms_ = 0.0;
    auto t_begin = std::chrono::steady_clock::now();

    while(ros::ok()){
        sensor_msgs::PointCloud2ConstPtr cloud_msg;
        if(!online_cloud_buffer_.WaitPop(cloud_msg, 0.5)){
            ros::spinOnce();
            // End of stream: at least one cloud received and nothing new for a while.
            if(online_idle_timeout_ > 0.0 && online_frame_cnt_ > 0){
                const double idle = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - last_online_msg_t_).count();
                if(idle > online_idle_timeout_){
                    mprtln(YELLOW, "[online] no cloud for %.1f s (online_idle_timeout = %.1f s), "
                            "the stream is over, wrapping up", idle, online_idle_timeout_);
                    return -2;
                }
            }
            t_begin = std::chrono::steady_clock::now();
            continue;
        }
        last_online_msg_t_ = std::chrono::steady_clock::now();
        if(!online_started_){ online_t_start_ = last_online_msg_t_; online_started_ = true; }
        online_wait_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - t_begin).count();
        online_wait_sum_ms_ += online_wait_ms_;

        // Same meaning as cloud_start_idx in the offline path.
        ++online_frame_cnt_;
        if(online_frame_cnt_ <= start_idx_){
            t_begin = std::chrono::steady_clock::now();
            continue;
        }

        cloud_time_ = cloud_msg->header.stamp.isZero() ? ros::Time::now() : cloud_msg->header.stamp;
        const auto p0 = std::chrono::steady_clock::now();
        frame_raw = ParseCloud2Msg(*cloud_msg, intensity_raw, data_set_);
        online_parse_ms_ = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - p0).count();
        if(frame_raw.first.empty()){
            mprtln(YELLOW, "[online warn] empty cloud on %s, frame skipped", semantic_cloud_topic_.c_str());
            return 0;
        }
        return 1;
    }
    return -1;
}

}   // namespace dero_slam
