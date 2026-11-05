#pragma once

#include <fstream>
#include <regex>
#include <filesystem>
// #include <pcl_conversions/pcl_conversions.h>
// #include <pcl/point_cloud.h>
// #include <pcl/point_types.h>

#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "deros_core/labels.hpp"
#include "dataprint.hpp"

#include <rosbag/bag.h>
#include <rosbag/view.h>



inline V3d_i LoadData(const std::string& bag_path, const std::string& lidar_topic, std::vector<float>& intensity_out, int& cloud_idx, const std::string& data_set) {
    V3d_i result;
    intensity_out.clear();

    bool enable_normalize = true;
    float intensity_max = 1.0f;
    float intensity_min = 0.0f;

    if (data_set == "mulran") {
        intensity_max = 716.0f;
    } else if (data_set == "kitti" || data_set == "kitti360") {
        intensity_max = 1.0f;
    } else if (data_set == "mid360") {
        intensity_max = 255.0f;
    } else if (data_set == "aeva") {
        intensity_min = -68.4073f;
        intensity_max = -36.9770f;
    } else if (data_set == "subt") {
        intensity_min = 1.0f;
        intensity_max = 100.0f;
    } else if (data_set == "geode") {
        intensity_min = 1.0f;
        intensity_max = 69.0f;
    }
     else {
        mprtln(YELLOW, "[warn] unknow data set type: %s, disable normalize", data_set.c_str());
        enable_normalize = false;
    }

    static rosbag::Bag bag;
    static std::shared_ptr<rosbag::View> lidar_view_ptr;
    static rosbag::View::iterator lidar_it;
    static bool initialized = false;

    if (!initialized) {
        try {
            bag.open(bag_path, rosbag::bagmode::Read);
        } catch (const std::exception& e) {
            ROS_ERROR("Failed to open bag %s: %s", bag_path.c_str(), e.what());
            return result;
        }

        lidar_view_ptr = std::make_shared<rosbag::View>(
            bag, rosbag::TopicQuery(lidar_topic));
        lidar_it = lidar_view_ptr->begin();
        initialized = true;

        int skipped = 0;
        while (lidar_it != lidar_view_ptr->end() && skipped < cloud_idx) {
            sensor_msgs::PointCloud2ConstPtr cloud_msg =
                lidar_it->instantiate<sensor_msgs::PointCloud2>();
            if (cloud_msg) {
                ++skipped;
            }
            ++lidar_it;
        }

        if (lidar_it == lidar_view_ptr->end()) {
            ROS_ERROR("No /lidar messages in bag or cloud_idx (%d) beyond total frames!",
                      cloud_idx);
            return result;
        }
    }

    while (lidar_it != lidar_view_ptr->end()) {
        sensor_msgs::PointCloud2ConstPtr cloud_msg =
            lidar_it->instantiate<sensor_msgs::PointCloud2>();
        if (cloud_msg) break;
        ++lidar_it;
    }

    if (lidar_it == lidar_view_ptr->end()) {
        ROS_INFO("End of bag reached.");
        return result;
    }

    sensor_msgs::PointCloud2ConstPtr cloud_msg =
        lidar_it->instantiate<sensor_msgs::PointCloud2>();

    if (!cloud_msg) {
        ++lidar_it;
        return result;
    }

    auto find_field = [&](const std::string& name) -> const sensor_msgs::PointField* {
        for (const auto& f : cloud_msg->fields) {
            if (f.name == name) return &f;
        }
        return nullptr;
    };

    const sensor_msgs::PointField* fx = find_field("x");
    const sensor_msgs::PointField* fy = find_field("y");
    const sensor_msgs::PointField* fz = find_field("z");

    if (!fx || !fy || !fz) {
        ROS_ERROR("PointCloud2 missing x/y/z fields!");
        ++lidar_it;
        return result;
    }

    const sensor_msgs::PointField* f_intensity = find_field("intensity");
    if (!f_intensity) {
        f_intensity = find_field("i");
    }

    const sensor_msgs::PointField* f_label = find_field("label");

    auto read_float = [](const uint8_t* ptr,
                         const sensor_msgs::PointField* f) -> float {
        if (!f) return 0.0f;

        switch (f->datatype) {
            case sensor_msgs::PointField::FLOAT32: {
                float v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return v;
            }
            case sensor_msgs::PointField::FLOAT64: {
                double v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<float>(v);
            }
            case sensor_msgs::PointField::UINT8: {
                uint8_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<float>(v);
            }
            case sensor_msgs::PointField::UINT16: {
                uint16_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<float>(v);
            }
            case sensor_msgs::PointField::UINT32: {
                uint32_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<float>(v);
            }
            case sensor_msgs::PointField::INT8: {
                int8_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<float>(v);
            }
            case sensor_msgs::PointField::INT16: {
                int16_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<float>(v);
            }
            case sensor_msgs::PointField::INT32: {
                int32_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<float>(v);
            }
            default:
                return 0.0f;
        }
    };

    auto read_u32 = [](const uint8_t* ptr,
                       const sensor_msgs::PointField* f) -> uint32_t {
        if (!f) return 0;

        switch (f->datatype) {
            case sensor_msgs::PointField::UINT32: {
                uint32_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return v;
            }
            case sensor_msgs::PointField::UINT16: {
                uint16_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<uint32_t>(v);
            }
            case sensor_msgs::PointField::UINT8: {
                uint8_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<uint32_t>(v);
            }
            case sensor_msgs::PointField::INT32: {
                int32_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<uint32_t>(v);
            }
            case sensor_msgs::PointField::INT16: {
                int16_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<uint32_t>(v);
            }
            case sensor_msgs::PointField::INT8: {
                int8_t v;
                std::memcpy(&v, ptr + f->offset, sizeof(v));
                return static_cast<uint32_t>(v);
            }
            default:
                return 0;
        }
    };

    V3d points;
    std::vector<int> labels;
    std::vector<float> intensities;

    const size_t num_points =
        static_cast<size_t>(cloud_msg->width) * cloud_msg->height;

    points.reserve(num_points);
    labels.reserve(num_points);
    intensities.reserve(num_points);

    const uint8_t* data_ptr = cloud_msg->data.data();
    const bool has_intensity_field = (f_intensity != nullptr);

    for (size_t row = 0; row < cloud_msg->height; ++row) {
        const uint8_t* row_ptr = data_ptr + row * cloud_msg->row_step;

        for (size_t col = 0; col < cloud_msg->width; ++col) {
            const uint8_t* ptr = row_ptr + col * cloud_msg->point_step;

            float x = read_float(ptr, fx);
            float y = read_float(ptr, fy);
            float z = read_float(ptr, fz);

            points.emplace_back(x, y, z);

            float raw_i = 0.0f;
            float norm_i = 0.0f;

            if (has_intensity_field) {
                raw_i = read_float(ptr, f_intensity);
                norm_i = raw_i;

                if (enable_normalize) {
                    const float range = intensity_max - intensity_min;
                    if (std::fabs(range) > 1e-6f) {
                        norm_i = (raw_i - intensity_min) / range;
                    }
                    norm_i = std::min(1.0f, std::max(0.0f, norm_i));
                }
            }
            intensities.push_back(norm_i);

            uint32_t raw_label = read_u32(ptr, f_label) & 0x0000ffff;
            auto it_map = label_map.find(static_cast<int>(raw_label));
            int mapped_label = (it_map != label_map.end()) ? it_map->second : 1;
            labels.push_back(mapped_label);
        }
    }

    result.first = std::move(points);
    result.second = std::move(labels);
    intensity_out = std::move(intensities);

    ++lidar_it;
    ++cloud_idx;

    return result;
}


inline std::string FixFrameId(const std::string &frame_id) {
    return std::regex_replace(frame_id, std::regex("^/"), "");
}

inline void addField(std::vector<sensor_msgs::PointField>& fields,
              const std::string& name, uint32_t offset,
              uint8_t datatype, uint32_t count) {
    sensor_msgs::PointField f;
    f.name = name;
    f.offset = offset;
    f.datatype = datatype;
    f.count = count;
    fields.push_back(f);
}


struct PointXYZL {
    float x, y, z;
    uint32_t label;
};

static_assert(sizeof(PointXYZL) == 16, "Unexpected padding");


struct PointXYZRGBL {
    float x, y, z;
    uint32_t label;
    uint32_t rgb;  // 0x00RRGGBB
};
static_assert(sizeof(PointXYZRGBL) == 20, "Unexpected padding");

inline sensor_msgs::PointCloud2 buildXYZRGBLCloud(const std::vector<Eigen::Vector4d>& points, const std::string& frame_id, const ros::Time& stamp) {
    sensor_msgs::PointCloud2 msg;
    msg.header.stamp = stamp;
    msg.header.frame_id = FixFrameId(frame_id);

    std::vector<PointXYZRGBL> packed;
    packed.reserve(points.size());
    for (const auto& pt : points) {
        int label = static_cast<int>(pt[3]);
        if (label == 0) continue;
        PointXYZRGBL p;
        p.x = static_cast<float>(pt[0]);
        p.y = static_cast<float>(pt[1]);
        p.z = static_cast<float>(pt[2]);
        p.label = static_cast<uint32_t>(label);
        uint8_t r = color_map[label][0];
        uint8_t g = color_map[label][1];
        uint8_t b = color_map[label][2];
        p.rgb = (static_cast<uint32_t>(r) << 16) | (static_cast<uint32_t>(g) << 8) | b;
        packed.push_back(p);
    }

    size_t n = packed.size();
    msg.height = 1;
    msg.width = n;
    msg.is_bigendian = false;
    msg.is_dense = true;

    std::vector<sensor_msgs::PointField> fields;
    int off = 0;
    addField(fields, "x", off, sensor_msgs::PointField::FLOAT32, 1); off += 4;
    addField(fields, "y", off, sensor_msgs::PointField::FLOAT32, 1); off += 4;
    addField(fields, "z", off, sensor_msgs::PointField::FLOAT32, 1); off += 4;
    addField(fields, "label", off, sensor_msgs::PointField::UINT32, 1); off += 4;
    addField(fields, "rgb", off, sensor_msgs::PointField::UINT32, 1); off += 4;

    msg.fields = fields;
    msg.point_step = off;
    msg.row_step = off * n;
    msg.data.resize(msg.row_step);

    std::memcpy(msg.data.data(), packed.data(), msg.row_step);
    return msg;
}
