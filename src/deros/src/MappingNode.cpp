#include "MappingNode.hpp"

#include <iomanip>
#include <sstream>

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <utility>

#include <cv_bridge/cv_bridge.h>

#include "OdometryNode.hpp"
#include "dataprint.hpp"
#include <filesystem>

namespace dero_slam{

namespace {

// SaveGlobalMap writes into <save_path>/map/, create it up front so that a
// missing directory cannot silently swallow the map.
void EnsureParentDir(const std::string &path) {
    std::error_code ec;
    const auto parent = std::filesystem::path(path).parent_path();
    if (!parent.empty()) std::filesystem::create_directories(parent, ec);
}

}   // namespace

Mapping::Mapping(const ros::NodeHandle& nh,const ros::NodeHandle& pnh):nh_(nh),pnh_(pnh){
    pnh_.param<std::string>("save_path", slam_mapping_cfg_.save_path, slam_mapping_cfg_.save_path);
    pnh_.param<std::string>("mapping_result_path",result_path_,result_path_);
    EnsureParentDir(result_path_);
    EnsureParentDir(slam_mapping_cfg_.save_path + "map/global_map.ply");
    run_mode_ = ReadRunMode(pnh_);
    pnh_.param<bool>("save_global_map",save_global_map_,true);
    pnh_.param<bool>("enable_eval",enable_eval_,true);
    pnh_.param<std::string>("odom_frame", odom_frame_, odom_frame_);
    pnh_.param<std::string>("child_frame", child_frame_, child_frame_);
    pnh_.param<std::string>("eval_cmd", eval_cmd_, eval_cmd_);
    pnh_.param<double>("max_range", slam_mapping_cfg_.max_range, slam_mapping_cfg_.max_range);
    pnh_.param<double>("min_range", slam_mapping_cfg_.min_range, slam_mapping_cfg_.min_range);
    pnh_.param<double>("bev_grid_size", slam_mapping_cfg_.bev_grid_size, slam_mapping_cfg_.bev_grid_size);
    pnh_.param<int>("close_kernel_size", slam_mapping_cfg_.close_kernel_size, slam_mapping_cfg_.close_kernel_size);
    pnh_.param<int>("loop_num_exclude", slam_mapping_cfg_.loop_num_exclude, slam_mapping_cfg_.loop_num_exclude);
    pnh_.param<int>("loop_num_search", slam_mapping_cfg_.loop_num_search, slam_mapping_cfg_.loop_num_search);
    pnh_.param<std::string>("loop_backend", slam_mapping_cfg_.loop_backend, slam_mapping_cfg_.loop_backend);
    pnh_.param<double>("max_expected_sigma", slam_mapping_cfg_.max_expected_sigma, slam_mapping_cfg_.max_expected_sigma);
    pnh_.param<int>("instance_threshold", slam_mapping_cfg_.instance_threshold, slam_mapping_cfg_.instance_threshold);
    pnh_.param<int>("keyframe_interval", slam_mapping_cfg_.keyframe_interval, slam_mapping_cfg_.keyframe_interval);
    pnh_.param<bool>("adaptive_weights", slam_mapping_cfg_.adaptive_weights, slam_mapping_cfg_.adaptive_weights);
    pnh_.param<int>("pgo_frame_interval", slam_mapping_cfg_.pgo_frame_interval, slam_mapping_cfg_.pgo_frame_interval);
    pnh_.param<bool>("enable_pgo", slam_mapping_cfg_.enable_pgo, slam_mapping_cfg_.enable_pgo);
    pnh_.param<double>("odom_rot_noise", slam_mapping_cfg_.odom_rot_noise, slam_mapping_cfg_.odom_rot_noise);
    pnh_.param<double>("odom_trans_noise", slam_mapping_cfg_.odom_trans_noise, slam_mapping_cfg_.odom_trans_noise);
    pnh_.param<double>("loop_rot_noise", slam_mapping_cfg_.loop_rot_noise, slam_mapping_cfg_.loop_rot_noise);
    pnh_.param<double>("loop_trans_noise", slam_mapping_cfg_.loop_trans_noise, slam_mapping_cfg_.loop_trans_noise);
    pnh_.param<int>("max_points_per_voxel", slam_mapping_cfg_.max_points_per_voxel, slam_mapping_cfg_.max_points_per_voxel);
    pnh_.param<double>("voxel_size",slam_mapping_cfg_.voxel_size,slam_mapping_cfg_.voxel_size);
    pnh_.param<int>("nFeatures",slam_mapping_cfg_.nFeatures,slam_mapping_cfg_.nFeatures);
    pnh_.param<float>("scaleFactor",slam_mapping_cfg_.scaleFactor,slam_mapping_cfg_.scaleFactor);
    pnh_.param<int>("nLevels",slam_mapping_cfg_.nLevels,slam_mapping_cfg_.nLevels);
    pnh_.param<int>("iniThFAST",slam_mapping_cfg_.iniThFAST,slam_mapping_cfg_.iniThFAST);
    pnh_.param<int>("minThFAST",slam_mapping_cfg_.minThFAST,slam_mapping_cfg_.minThFAST);
    pnh_.param<double>("out_dists_th",slam_mapping_cfg_.out_dists_th,slam_mapping_cfg_.out_dists_th);
    pnh_.param<double>("traj_line_width", traj_line_width_, traj_line_width_);
    


    slam_mapping_ = deros_api::SlamMapping(slam_mapping_cfg_);

    traj_pgo_publisher_ = pnh_.advertise<nav_msgs::Path>("trajectory_pgo", 1);
    traj_grad_publisher_ = pnh_.advertise<visualization_msgs::Marker>("trajectory_gradient", 1);
    loop_edge_publisher_ = pnh_.advertise<visualization_msgs::MarkerArray>("loop_edges", 1);
    global_map_pub_ = pnh_.advertise<sensor_msgs::PointCloud2>("global_map", 1);
    ground_line_pub_ = pnh_.advertise<sensor_msgs::Image>("ground_line_image", 1);
    building_pub_ = pnh_.advertise<sensor_msgs::Image>("building_image", 1);
    debug_info_pub_ = pnh_.advertise<sensor_msgs::Image>("debug_info", 1);
    fout_mapping_.open(result_path_,std::ofstream::trunc);
    fout_mapping_.close(); // clear the file

}

Mapping::~Mapping(){}



void Mapping::ScanMapping(){
    ros::Rate loop(10);
    while(ros::ok()){
        if(scan_queue.empty()){
            if(break_sign){
                break;
            }
            continue;
        }


        auto t1 = std::chrono::steady_clock::now();
        scan_mutex.lock();
        auto frame_data = std::move(scan_queue.front());
        scan_queue.pop();
        scan_mutex.unlock();


        deros_api::MappingRes mapping_res;

        double trans = 0.0;
        Sophus::SE3d relative_pose;
        if (last_cloud_idx_ >= 0) {
            relative_pose = last_pose_.inverse() * frame_data.pose;
            trans = relative_pose.translation().norm();
        }
        cul_ref_odom_ *= relative_pose;

        const Sophus::SE3d cur_opt_pose = latest_opt_pose_ * cul_ref_odom_;
        if (odom_) odom_->SetVizPose(cur_opt_pose);

        if (loop_skip_counter_ > 0 && trans > 0.3)
            loop_skip_counter_--;

        bool allow_detection = (loop_skip_counter_ == 0) && (last_cloud_idx_ >= 0);

        if (allow_detection) {
            mapping_res = slam_mapping_.mainProcess(
                frame_data.frame, 
                cur_opt_pose, 
                frame_data.intensity, 
                frame_data.model_deviation, 
                frame_data.ground_bev_grid, 
                frame_data.noground_bev_grid, 
                frame_data.cloud_id);
        }



        mapping_mutex.lock();
        mapping_message.gnd_line_count = mapping_res.gnd_line_count;
        mapping_message.gnd_its_count = mapping_res.gnd_its_count;
        mapping_message.building_orient_bins = mapping_res.building_orient_bins;
        mapping_message.building_rho2 = mapping_res.building_rho2;
        mapping_message.building_line_len = mapping_res.building_line_len;
        mapping_mutex.unlock();

        // bool is_loop = (frame_data.cloud_id % slam_mapping_cfg_.keyframe_interval == 0);
        // if (is_loop) {
        //     mapping_res = slam_mapping_.mainProcess(
        //         frame_data.frame, 
        //         frame_data.pose, 
        //         frame_data.model_deviation, 
        //         frame_data.cloud_id);
        // }

        auto t2 = std::chrono::steady_clock::now();

        if(slam_mapping_cfg_.enable_pgo){
            pose_graph_.SetInitialPose(frame_data.cloud_id, frame_data.pose);

            if (last_cloud_idx_ >= 0) {
                pose_graph_.AddOdometryEdge(
                    last_cloud_idx_, 
                    frame_data.cloud_id, 
                    relative_pose, 
                    slam_mapping_cfg_.odom_rot_noise, 
                    slam_mapping_cfg_.odom_trans_noise);
            } else {
                pose_graph_.AddPriorFactor(frame_data.cloud_id, frame_data.pose);
            }

            if (allow_detection && mapping_res.loop.is_valid) {
                pose_graph_.AddLoopClosureEdge(
                    mapping_res.loop.match_idx, 
                    mapping_res.loop.curr_idx, 
                    mapping_res.loop.relative_pose,
                    mapping_res.loop.rot_noise_scale * slam_mapping_cfg_.loop_rot_noise,
                    mapping_res.loop.trans_noise_scale * slam_mapping_cfg_.loop_trans_noise);
                // pose_graph_.AddLoopClosureEdge(
                //     mapping_res.loop.match_idx, 
                //     mapping_res.loop.curr_idx, 
                //     mapping_res.loop.relative_pose,
                //     slam_mapping_cfg_.loop_rot_noise,
                //     slam_mapping_cfg_.loop_trans_noise);

                ROS_INFO("Loop closure between %d and %d |scale: %.3f", mapping_res.loop.curr_idx, mapping_res.loop.match_idx, mapping_res.loop.trans_noise_scale);
                loop_pair_vec_.emplace_back(mapping_res.loop.curr_idx, mapping_res.loop.match_idx);
                loop_skip_counter_ = slam_mapping_cfg_.keyframe_interval;
            }
            // if (mapping_res.loop.is_valid) {
            //     // pose_graph_.AddLoopClosureEdge(
            //     //     mapping_res.loop.match_idx, 
            //     //     mapping_res.loop.curr_idx, 
            //     //     mapping_res.loop.relative_pose,
            //     //     mapping_res.loop.rot_noise_scale * slam_mapping_cfg_.loop_rot_noise,
            //     //     mapping_res.loop.trans_noise_scale * slam_mapping_cfg_.loop_trans_noise);
            //     pose_graph_.AddLoopClosureEdge(
            //         mapping_res.loop.match_idx, 
            //         mapping_res.loop.curr_idx, 
            //         mapping_res.loop.relative_pose,
            //         slam_mapping_cfg_.loop_rot_noise,
            //         slam_mapping_cfg_.loop_trans_noise);
            //     ROS_INFO("Loop closure between %d and %d |scale: %.3f", mapping_res.loop.curr_idx, mapping_res.loop.match_idx, mapping_res.loop.trans_noise_scale);
            //     loop_pair_vec_.emplace_back(mapping_res.loop.curr_idx, mapping_res.loop.match_idx);
            //     loop_skip_counter_ = slam_mapping_cfg_.keyframe_interval;
            // }
            last_cloud_idx_ = frame_data.cloud_id;
            last_pose_ = frame_data.pose;

            if (++frame_counter_ >= slam_mapping_cfg_.pgo_frame_interval) {
                frame_counter_ = 0;
                pose_graph_.Optimize();
                poses_pgo_ = pose_graph_.GetAllOptimizedPoses(last_cloud_idx_);
                latest_opt_pose_ = pose_graph_.GetOptimizedPose(last_cloud_idx_);

                slam_mapping_.UpdateAllPoseDescriptors(poses_pgo_);
                cul_ref_odom_ = Sophus::SE3d();
                PublishTrajectory();
                visualizeLoopClosure();
            }
        }


        auto t3 = std::chrono::steady_clock::now();
        PublishImage(ground_line_pub_ ,mapping_res.ground_line_in_map, ros::Time::now(), odom_frame_);
        PublishImage(building_pub_ ,mapping_res.building_line_in_map, ros::Time::now(), odom_frame_);
        DebugImageInfo dbg;
        dbg.dev_flag = frame_data.deviation_flag;
        dbg.dev_score = frame_data.deviation_score;
        dbg.gnd_line = mapping_res.gnd_line_count;
        dbg.gnd_its = mapping_res.gnd_its_count;
        dbg.ground_ratio = frame_data.ground_ratio;
        dbg.ground_score = frame_data.deviation_score_g;
        dbg.building_rho2 = mapping_res.building_rho2;
        dbg.building_len = mapping_res.building_line_len;
        dbg.building_score = frame_data.deviation_score_b;
        cv::Mat debug_img = CreateDebugImage(dbg);
        PublishImage(debug_info_pub_, debug_img, ros::Time::now(), odom_frame_);
        if (run_mode_ != run_mode::kRosRun) {
            {
                std::lock_guard<std::mutex> lock(sync_mutex);
                frame_processed = true;
            }
            sync_cv.notify_one();
        }

        if(run_mode_==run_mode::kEvalOnly || run_mode_==run_mode::kEvalAndRos || run_mode_==run_mode::kOnline){
            // mprtln(MAGENTA,"[Mapping] cloud_idx: %d | main_time:%.3f ms | all_time:%.3f ms\r\n",frame_data.cloud_id, (float)(t2-t1).count()*1e-6, (float)(t3-t1).count()*1e-6);
            ros::spinOnce();
            continue;
        }
        mprtln(MAGENTA,"[Mapping] cloud_idx: %d | main_time:%.3f ms | all_time:%.3f ms\r\n",frame_data.cloud_id, (float)(t2-t1).count()*1e-6, (float)(t3-t1).count()*1e-6);

        loop.sleep();
        ros::spinOnce();
    }
    SaveOptimizedTrajectory();
    PublishGlobalMap();
    EvoEval();
}

void Mapping::PublishImage(const ros::Publisher& pub, const cv::Mat& image, const ros::Time& stamp, const std::string& frame_id) {
    if (pub.getNumSubscribers() == 0 || image.empty())
        return;

    cv::Mat img_to_pub;
    if (image.type() == CV_8UC3) {
        img_to_pub = image;
    } else if (image.type() == CV_8UC1) {
        cv::cvtColor(image, img_to_pub, cv::COLOR_GRAY2BGR);
    } else {
        ROS_WARN("image type not supported, skip publishing.");
        return;
    }

    sensor_msgs::ImagePtr msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", img_to_pub).toImageMsg();
    msg->header.stamp = stamp;
    msg->header.frame_id = frame_id;
    pub.publish(msg);
}

cv::Mat Mapping::CreateDebugImage(const DebugImageInfo &info) {
    const int width = 620;
    const int height = 400;
    cv::Mat img(height, width, CV_8UC3, cv::Scalar(30, 30, 30));

    const cv::Scalar white(255, 255, 255);
    const cv::Scalar grey(110, 110, 110);
    const cv::Scalar green(120, 255, 120);
    const cv::Scalar red(90, 90, 255);
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = 0.62;
    const int thickness = 1;

    int y = 32;
    auto put = [&](const std::string &text, const cv::Scalar &color) {
        cv::putText(img, text, cv::Point(16, y), font_face, font_scale, color, thickness);
        y += 30;
    };
    // Horizontal separator between the deviation summary, the ground block and
    // the building block.
    auto separator = [&]() {
        y -= 10;
        cv::line(img, cv::Point(12, y), cv::Point(width - 12, y), grey, 1);
        y += 28;
    };
    auto fmt = [](double v, int digits) {
        std::ostringstream os;
        os << std::fixed << std::setprecision(digits) << v;
        return os.str();
    };

    put("Deviation Flag: " + std::string(info.dev_flag ? "YES" : "NO"), info.dev_flag ? red : green);
    put("Deviation Score: " + fmt(info.dev_score, 3), white);
    separator();
    put("Ground Line Count: " + std::to_string(info.gnd_line), white);
    put("Ground ITS Count: " + std::to_string(info.gnd_its), white);
    put("Ground Point Ratio: " + fmt(info.ground_ratio, 3), white);
    put("Ground Ratio Score: " + fmt(info.ground_score, 3), white);
    separator();
    put("Building Line Length: " + fmt(info.building_len, 1), white);
    put("Building Rho2: " + fmt(info.building_rho2, 3), white);
    put("Building Score: " + fmt(info.building_score, 3), white);

    return img;
}

void Mapping::PublishGlobalMap() {
    const auto& [indices, frames] = slam_mapping_.ReturnAllFrames();
    if (poses_pgo_.empty() || frames.empty()) {
        mprtln(YELLOW, "[GlobalMap] no optimized pose yet, global map skipped");
        return;
    }

    const auto& poses = poses_pgo_;
    const int step = 5;
    const double voxel_size = 0.3;

    // Publishing only makes sense when someone listens (rviz); saving is independent.
    if (global_map_pub_.getNumSubscribers() != 0) {
        for (size_t i = 0; i < frames.size(); i += step) {
            int cloud_idx = indices[i];
            if (cloud_idx < 0 || cloud_idx >= poses.size()) continue;
            const Sophus::SE3d& Twc = poses[cloud_idx];
            if (cloud_idx > 0 && Twc.matrix().isIdentity(1e-10)) continue;

            V3d_i frame_down = deros_api::DownsampleMean(frames[i], voxel_size);
            V4d points;
            points.reserve(frame_down.first.size());
            for (size_t j = 0; j < frame_down.first.size(); ++j) {
                Eigen::Vector3d pw = Twc * frame_down.first[j];
                points.emplace_back(pw.x(), pw.y(), pw.z(), static_cast<double>(frame_down.second[j]));
            }

            sensor_msgs::PointCloud2 msg = buildXYZRGBLCloud(points, odom_frame_, ros::Time::now());
            global_map_pub_.publish(msg);
            ros::spinOnce();
        }
        mprtln(GREEN, "[GlobalMap] published (subscribers = %d)", global_map_pub_.getNumSubscribers());
    }

    if(save_global_map_){
        SaveGlobalMap(slam_mapping_cfg_.save_path + "map/global_map.ply", slam_mapping_cfg_.save_path + "map/global_labels.txt", voxel_size, step);
        mprtln(GREEN, "[GlobalMap] saved to %smap/global_map.ply", slam_mapping_cfg_.save_path.c_str());
    }
}

void Mapping::WritePLY(const std::string& file_path,
                       const std::vector<Eigen::Vector3d>& points,
                       const std::vector<int>& labels,
                       const std::unordered_map<int, Eigen::Vector3i>& color_map) {
    if (points.empty()) {
        std::cerr << "No points to save!" << std::endl;
        return;
    }
    size_t num_vertices = points.size();
    std::ofstream ofs(file_path, std::ios::binary);
    if (!ofs) {
        std::cerr << "Cannot open file: " << file_path << std::endl;
        return;
    }
    ofs << "ply\n";
    ofs << "format binary_little_endian 1.0\n";
    ofs << "element vertex " << num_vertices << "\n";
    ofs << "property float x\n";
    ofs << "property float y\n";
    ofs << "property float z\n";
    ofs << "property uchar red\n";
    ofs << "property uchar green\n";
    ofs << "property uchar blue\n";
    ofs << "property int label\n";
    ofs << "end_header\n";

    for (size_t i = 0; i < num_vertices; ++i) {
        const auto& pt = points[i];
        int label = labels[i];
        Eigen::Vector3i color(128,128,128);
        auto it = color_map.find(label);
        if (it != color_map.end()) {
            color = it->second;
        }
        float x = static_cast<float>(pt.x());
        float y = static_cast<float>(pt.y());
        float z = static_cast<float>(pt.z());
        unsigned char r = static_cast<unsigned char>(color.x());
        unsigned char g = static_cast<unsigned char>(color.y());
        unsigned char b = static_cast<unsigned char>(color.z());
        int lbl = label;
        ofs.write(reinterpret_cast<const char*>(&x), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&y), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&z), sizeof(float));
        ofs.write(reinterpret_cast<const char*>(&r), sizeof(unsigned char));
        ofs.write(reinterpret_cast<const char*>(&g), sizeof(unsigned char));
        ofs.write(reinterpret_cast<const char*>(&b), sizeof(unsigned char));
        ofs.write(reinterpret_cast<const char*>(&lbl), sizeof(int));
    }
    ofs.close();
}

void Mapping::SaveGlobalMap(const std::string& ply_path,
                            const std::string& label_path,
                            double voxel_size,
                            int step) {
    const auto& [indices, frames] = slam_mapping_.ReturnAllFrames();
    if (poses_pgo_.empty()) {
        std::cerr << "Poses empty, cannot save map." << std::endl;
        return;
    }
    const auto& poses = poses_pgo_;

    std::vector<Eigen::Vector3d> all_points;
    std::vector<int> all_labels;
    mprtln(GREEN,"frames.size() : %d",frames.size());
    for (size_t i = 0; i < frames.size(); i += step) {
        int cloud_idx = indices[i];
        if (cloud_idx < 0 || cloud_idx >= static_cast<int>(poses.size())) continue;
        const Sophus::SE3d& Twc = poses[cloud_idx];
        if (cloud_idx > 0 && Twc.matrix().isIdentity(1e-10)) continue;

        V3d_i frame_down = deros_api::DownsampleMean(frames[i], voxel_size);
        if (frame_down.first.empty()) continue;

        for (size_t j = 0; j < frame_down.first.size(); ++j) {
            Eigen::Vector3d pw = Twc * frame_down.first[j];
            all_points.emplace_back(pw);
            all_labels.emplace_back(frame_down.second[j]);
        }
    }

    if (all_points.empty()) {
        std::cerr << "No points collected, save failed." << std::endl;
        return;
    }

    WritePLY(ply_path, all_points, all_labels, color_map);

    std::ofstream lf(label_path);
    if (!lf) {
        std::cerr << "Cannot open label file: " << label_path << std::endl;
    } else {
        for (int lbl : all_labels) {
            lf << lbl << "\n";
        }
        lf.close();
    }
    std::cout << "Global map saved to: " << ply_path << std::endl;
    std::cout << "Labels saved to: " << label_path << std::endl;
}



void Mapping::SaveOptimizedTrajectory() {
    pose_graph_.Optimize();

    poses_pgo_  = pose_graph_.GetAllOptimizedPoses(last_cloud_idx_);

    const Sophus::SE3d T_first = poses_pgo_[1];
    
    fout_mapping_.open(result_path_, std::ofstream::trunc);
    if (!fout_mapping_.is_open()) {
        mprtln(YELLOW, "Cannot open PGO result file: %s", result_path_.c_str());
        return;
    }
    fout_mapping_ << std::fixed << std::setprecision(16);
    for (size_t i = 1; i < poses_pgo_.size(); ++i) {
        
        const auto& opt_pose = T_first.inverse() * poses_pgo_[i];
        // if (i > 0 && opt_pose.matrix().isIdentity(1e-10)) continue;

        Eigen::Matrix4d mat = opt_pose.matrix();
        fout_mapping_   << mat(0,0) << " " << mat(0,1) << " " << mat(0,2) << " " << mat(0,3) << " "
                        << mat(1,0) << " " << mat(1,1) << " " << mat(1,2) << " " << mat(1,3) << " "
                        << mat(2,0) << " " << mat(2,1) << " " << mat(2,2) << " " << mat(2,3) << std::endl;
    }
    fout_mapping_.close();
}

namespace {
inline void TrajGradientColor(float t, float& r, float& g, float& b) {
    t = std::max(0.0f, std::min(1.0f, t));
    const float h = 0.66f * (1.0f - t);
    const float sector = h * 6.0f;
    const int   sector_i = static_cast<int>(std::floor(sector)) % 6;
    const float f = sector - std::floor(sector);
    const float p = 0.0f;
    const float q = 1.0f - f;
    const float u = f;
    switch (sector_i) {
        case 0: r = 1.0f; g = u;    b = p;    break;
        case 1: r = q;    g = 1.0f; b = p;    break;
        case 2: r = p;    g = 1.0f; b = u;    break;
        case 3: r = p;    g = q;    b = 1.0f; break;
        case 4: r = u;    g = p;    b = 1.0f; break;
        default: r = 1.0f; g = p;   b = q;    break;
    }
}
}  // namespace

void Mapping::PublishTrajectory() {
    if (poses_pgo_.empty()) return;

    const bool want_path = traj_pgo_publisher_.getNumSubscribers() > 0;
    const bool want_grad = traj_grad_publisher_.getNumSubscribers() > 0;
    if (!want_path && !want_grad) return;

    const ros::Time stamp = ros::Time::now();

    nav_msgs::Path path_msg;
    path_msg.header.frame_id = odom_frame_;
    path_msg.header.stamp = stamp;

    visualization_msgs::Marker grad;
    grad.header.frame_id = odom_frame_;
    grad.header.stamp = stamp;
    grad.ns = "trajectory_gradient";
    grad.id = 0;
    grad.type = visualization_msgs::Marker::LINE_STRIP;
    grad.action = visualization_msgs::Marker::ADD;
    grad.pose.orientation.w = 1.0;
    grad.scale.x = traj_line_width_;
    grad.color.a = 1.0;

    const int last = std::min(last_cloud_idx_, static_cast<int>(poses_pgo_.size()) - 1);
    const float denom = static_cast<float>(std::max(1, last));

    for (int i = 0; i <= last; ++i) {
        const auto& opt_pose = poses_pgo_[i];
        if (opt_pose.matrix().isIdentity(1e-10) && i > 0) continue;

        const Eigen::Vector3d t = opt_pose.translation();
        if (want_path) {
            geometry_msgs::PoseStamped pose_stamped;
            pose_stamped.header.frame_id = odom_frame_;
            pose_stamped.header.stamp = stamp;
            pose_stamped.pose.position.x = t.x();
            pose_stamped.pose.position.y = t.y();
            pose_stamped.pose.position.z = t.z();
            Eigen::Quaterniond q(opt_pose.rotationMatrix());
            pose_stamped.pose.orientation.x = q.x();
            pose_stamped.pose.orientation.y = q.y();
            pose_stamped.pose.orientation.z = q.z();
            pose_stamped.pose.orientation.w = q.w();
            path_msg.poses.push_back(pose_stamped);
        }

        if (want_grad) {
            geometry_msgs::Point pt;
            pt.x = t.x();
            pt.y = t.y();
            pt.z = t.z();
            grad.points.push_back(pt);

            std_msgs::ColorRGBA c;
            c.a = 1.0f;
            TrajGradientColor(static_cast<float>(i) / denom, c.r, c.g, c.b);
            grad.colors.push_back(c);
        }
    }

    if (want_path) traj_pgo_publisher_.publish(path_msg);
    if (want_grad && grad.points.size() >= 2) traj_grad_publisher_.publish(grad);
}

void Mapping::visualizeLoopClosure() {
    if (loop_edge_publisher_.getNumSubscribers() == 0)
        return;

    if (loop_pair_vec_.empty() || poses_pgo_.empty()) return;

    visualization_msgs::MarkerArray marker_array;

    auto create_marker = [&](int id, const std::string& ns, int type, float r, float g, float b, float scale) {
        visualization_msgs::Marker m;
        m.header.frame_id = odom_frame_;
        m.header.stamp = ros::Time::now();
        m.ns = ns;
        m.id = id;
        m.type = type;
        m.action = visualization_msgs::Marker::ADD;
        m.pose.orientation.w = 1.0;
        m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0;
        m.scale.x = scale;
        if (type == visualization_msgs::Marker::LINE_LIST)
            m.scale.y = m.scale.z = 0.0;
        else if (type == visualization_msgs::Marker::SPHERE_LIST)
            m.scale.y = m.scale.z = scale;
        m.lifetime = ros::Duration(0);
        return m;
    };

    auto edge_marker = create_marker(0, "loop_edges", visualization_msgs::Marker::LINE_LIST, 0.0, 1.0, 0.0, 0.1);
    auto node_marker = create_marker(1, "loop_nodes", visualization_msgs::Marker::SPHERE_LIST, 1.0, 0.0, 0.0, 1.5);

    for (const auto& info : loop_pair_vec_) {
        if (info.first >= poses_pgo_.size() || info.second >= poses_pgo_.size())
            continue;

        const Eigen::Vector3d& p1 = poses_pgo_[info.first].translation();
        const Eigen::Vector3d& p2 = poses_pgo_[info.second].translation();

        geometry_msgs::Point pt1, pt2;
        pt1.x = p1.x(); pt1.y = p1.y(); pt1.z = p1.z();
        pt2.x = p2.x(); pt2.y = p2.y(); pt2.z = p2.z();

        edge_marker.points.push_back(pt1);
        edge_marker.points.push_back(pt2);
        node_marker.points.push_back(pt1);
        node_marker.points.push_back(pt2);
    }

    marker_array.markers.push_back(edge_marker);
    marker_array.markers.push_back(node_marker);
    loop_edge_publisher_.publish(marker_array);
}



void Mapping::EvoEval() const {
    if (!enable_eval_ || eval_cmd_.empty()) return;

    ROS_INFO_STREAM("Running evaluation: " << eval_cmd_);

    int ret = std::system(eval_cmd_.c_str());
    if (ret != 0) {
        ROS_ERROR_STREAM("Evaluation command failed with code: " << ret);
    }
}

}   // namespace dero_slam
