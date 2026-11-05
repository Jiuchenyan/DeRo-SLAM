#include <memory>
#include <thread>

#include "MappingNode.hpp"
#include "OdometryNode.hpp"

int main(int argc, char** argv){

    ros::init(argc, argv, "deros_node");
    ros::NodeHandle nh;
    ros::NodeHandle nh_private("~");

    // In the online mode the front end blocks until the next cloud arrives, so
    // the subscription callbacks need their own thread: with the plain
    // spinOnce() calls they would only be served once per processed frame and
    // the subscription queue would drop clouds.
    int run_mode = 0;
    nh_private.param<int>("run_mode", run_mode, 0);
    if (!nh_private.hasParam("run_mode")) nh_private.param<int>("debug_mode", run_mode, 0);
    std::unique_ptr<ros::AsyncSpinner> spinner;
    if (run_mode == dero_slam::run_mode::kOnline) {
        spinner.reset(new ros::AsyncSpinner(2));
        spinner->start();
        mprtln(GREEN, "[online] async spinner started (2 threads)");
    }

    ros::Rate init_wait(0.5);
    dero_slam::Odometry odom(nh,nh_private);
    dero_slam::Mapping map(nh,nh_private);
    map.SetOdometry(&odom);
    mprtln(GREEN,"waiting for initialization");
    init_wait.sleep();
    std::thread odometry_thread(&dero_slam::Odometry::RegisterFrame,&odom);
    std::thread mapping_thread(&dero_slam::Mapping::ScanMapping,&map);

    odometry_thread.join();
    mapping_thread.join();

    return 0;
}
