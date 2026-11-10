cmake_minimum_required(VERSION 3.16)
project(deros)

# ---------------------------------------------------------------------------
# ROS application layer of DeRo-SLAM.
#
# The algorithms themselves live in ../deros_core (prebuilt shared library +
# public headers).  This package only contains the ROS glue: reading the bag,
# publishing topics, running the pose graph and writing the results.
# ---------------------------------------------------------------------------

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs nav_msgs sensor_msgs roscpp rosbag std_msgs cv_bridge tf tf2 tf2_ros)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui features2d calib3d)
# libderos_core.so is dynamically linked against libgtsam.so.4, which was built
# from GTSAM 4.2.0 with TBB. The same ABI has to be visible at link time.
find_package(GTSAM QUIET)
if(NOT GTSAM_FOUND)
  find_library(GTSAM_LIBRARY NAMES gtsam
               PATHS /usr/local/lib /usr/lib/x86_64-linux-gnu /usr/lib)
endif()
if(NOT GTSAM_FOUND AND NOT GTSAM_LIBRARY)
  message(FATAL_ERROR
    "GTSAM not found.\n"
    "libderos_core.so is linked against libgtsam.so.4, built from GTSAM 4.2.0 with TBB\n"
    "(GTSAM_USE_TBB=ON). A different version, or a build without TBB, fails at link\n"
    "time with undefined references to gtsam::* symbols.\n"
    "Install it as described in BUILD.md, or point CMake at an existing build:\n"
    "  catkin_make -DGTSAM_DIR=/path/to/gtsam/lib/cmake/GTSAM\n"
    "  catkin_make -DGTSAM_LIBRARY=/path/to/libgtsam.so.4")
endif()
catkin_package()

set(DEROS_CORE_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../deros_core")
set(DEROS_CORE_LIB  "${DEROS_CORE_ROOT}/lib/libderos_core.so")
if(NOT EXISTS "${DEROS_CORE_LIB}")
  message(FATAL_ERROR
    "libderos_core.so not found at ${DEROS_CORE_LIB}.\n"
    "Run ./package.sh in the workspace root to (re)build the core library.")
endif()

add_executable(deros_node
  src/deros_rosnode.cpp
  src/OdometryNode.cpp
  src/MappingNode.cpp
  src/NodeShared.cpp)

target_include_directories(deros_node PRIVATE
  include
  ${DEROS_CORE_ROOT}/include
  ${DEROS_CORE_ROOT}/include/third_party/eigen
  ${DEROS_CORE_ROOT}/include/third_party/sophus
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS})

target_link_libraries(deros_node PRIVATE
  ${DEROS_CORE_LIB}
  ${OpenCV_LIBS}
  ${catkin_LIBRARIES})

if(GTSAM_FOUND)
  target_link_libraries(deros_node PRIVATE gtsam)
elseif(GTSAM_LIBRARY)
  target_link_libraries(deros_node PRIVATE ${GTSAM_LIBRARY})
endif()

# Sophus is header only and, without this switch, drags fmt in. The library is
# built with the same setting, so the inline code agrees between the two sides.
target_compile_definitions(deros_node PRIVATE SOPHUS_USE_BASIC_LOGGING=1)

# libderos_core.so sits next to the package, not in a system path.
set_target_properties(deros_node PROPERTIES
  BUILD_RPATH "${DEROS_CORE_ROOT}/lib"
  INSTALL_RPATH "${DEROS_CORE_ROOT}/lib")
