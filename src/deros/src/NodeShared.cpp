#include "NodeShared.hpp"

#include "ros/node_handle.h"
#include "ros/console.h"

#include "dataprint.hpp"

namespace dero_slam {

std::queue<scan_data> scan_queue;
std::mutex scan_mutex;

mapping_data mapping_message;
std::mutex mapping_mutex;

std::mutex sync_mutex;
std::condition_variable sync_cv;
bool frame_processed = true;
bool break_sign = false;

const char* RunModeName(int mode) {
    switch (mode) {
        case run_mode::kRosRun:     return "ROS_RUN";
        case run_mode::kEvalOnly:   return "EVAL_ONLY";
        case run_mode::kEvalAndRos: return "EVAL_AND_ROS";
        case run_mode::kOnline:     return "ONLINE_SEMANTIC";
        default:                    return "UNKNOWN";
    }
}

int ReadRunMode(const ros::NodeHandle &pnh) {
    int mode = run_mode::kRosRun;
    if (pnh.hasParam("run_mode")) {
        pnh.param<int>("run_mode", mode, mode);
    } else if (pnh.hasParam("debug_mode")) {
        pnh.param<int>("debug_mode", mode, mode);
        mprtln(YELLOW, "[warn] parameter debug_mode has been renamed to run_mode (value %d)", mode);
    }
    return mode;
}

}   // namespace dero_slam
