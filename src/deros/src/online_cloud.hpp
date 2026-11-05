#pragma once

/**
 * @file online_cloud.hpp
 * @brief 在线（ROS 话题）点云输入：订阅语义推理节点发布的 PointCloud2，
 *        解析成与离线 LoadData() 完全相同的 (V3d points, std::vector<int> labels) + intensity。
 *
 * 设计原则：
 *  - 完全独立于 dataprocess.hpp 中的离线 LoadData()，离线路径一行未改；
 *  - 解析规则（字段选择、强度归一化、label_map 映射）与 LoadData() 保持一致，
 *    这样在线模式与离线模式喂给 SlamOdometry::mainProcess() 的数据语义相同。
 *
 * 点云字段约定（语义推理节点发布的话题）：
 *   x, y, z            FLOAT32   —— 必需
 *   intensity / i      FLOAT32   —— 可选，缺省为 0
 *   label 或 sem_label UINT8/16  —— 可选，原始 SemanticKITTI 标签（0,10,...,259），
 *                                  经 label_map 映射成 DeRo 内部类别；缺省为 0（unlabeled）
 *   其余字段（ring/time/prob/color...）一律忽略。
 */

#include <algorithm>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <vector>

#include "sensor_msgs/PointCloud2.h"

#include "deros_core/labels.hpp"
#include "dataprint.hpp"

namespace dero_slam {

// 强度归一化参数（与 dataprocess.hpp::LoadData 中的表保持一致）
struct IntensityNormParam {
    bool enable = true;
    float min_v = 0.0f;
    float max_v = 1.0f;
};

inline IntensityNormParam GetIntensityNormParam(const std::string& data_set) {
    IntensityNormParam p;
    if (data_set == "mulran") {
        p.max_v = 716.0f;
    } else if (data_set == "kitti" || data_set == "kitti360") {
        p.max_v = 1.0f;
    } else if (data_set == "mid360") {
        p.max_v = 255.0f;
    } else if (data_set == "aeva") {
        p.min_v = -68.4073f;
        p.max_v = -36.9770f;
    } else if (data_set == "subt") {
        p.min_v = 1.0f;
        p.max_v = 100.0f;
    } else if (data_set == "geode") {
        p.min_v = 1.0f;
        p.max_v = 69.0f;
    } else {
        mprtln(YELLOW, "[online warn] unknow data set type: %s, disable normalize", data_set.c_str());
        p.enable = false;
    }
    return p;
}

namespace detail {

inline const sensor_msgs::PointField* FindField(const sensor_msgs::PointCloud2& msg,
                                                const std::string& name) {
    for (const auto& f : msg.fields) {
        if (f.name == name) return &f;
    }
    return nullptr;
}

inline float ReadFloat(const uint8_t* ptr, const sensor_msgs::PointField* f) {
    if (!f) return 0.0f;
    switch (f->datatype) {
        case sensor_msgs::PointField::FLOAT32: { float v; std::memcpy(&v, ptr + f->offset, sizeof(v)); return v; }
        case sensor_msgs::PointField::FLOAT64: { double v; std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<float>(v); }
        case sensor_msgs::PointField::UINT8:   { uint8_t v;  std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<float>(v); }
        case sensor_msgs::PointField::UINT16:  { uint16_t v; std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<float>(v); }
        case sensor_msgs::PointField::UINT32:  { uint32_t v; std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<float>(v); }
        case sensor_msgs::PointField::INT8:    { int8_t v;   std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<float>(v); }
        case sensor_msgs::PointField::INT16:   { int16_t v;  std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<float>(v); }
        case sensor_msgs::PointField::INT32:   { int32_t v;  std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<float>(v); }
        default: return 0.0f;
    }
}

inline uint32_t ReadUInt(const uint8_t* ptr, const sensor_msgs::PointField* f) {
    if (!f) return 0;
    switch (f->datatype) {
        case sensor_msgs::PointField::UINT32: { uint32_t v; std::memcpy(&v, ptr + f->offset, sizeof(v)); return v; }
        case sensor_msgs::PointField::UINT16: { uint16_t v; std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<uint32_t>(v); }
        case sensor_msgs::PointField::UINT8:  { uint8_t v;  std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<uint32_t>(v); }
        case sensor_msgs::PointField::INT32:  { int32_t v;  std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<uint32_t>(v); }
        case sensor_msgs::PointField::INT16:  { int16_t v;  std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<uint32_t>(v); }
        case sensor_msgs::PointField::INT8:   { int8_t v;   std::memcpy(&v, ptr + f->offset, sizeof(v)); return static_cast<uint32_t>(v); }
        default: return 0;
    }
}

}  // namespace detail

/**
 * @brief 把一帧 PointCloud2 解析成 DeRo 前端需要的形式（与离线 LoadData 同语义）
 * @param msg            订阅到的点云
 * @param intensity_out  输出：归一化后的强度
 * @param data_set       数据集名，用于强度归一化区间
 * @return (points, labels)；解析失败时 points 为空
 */
inline V3d_i ParseCloud2Msg(const sensor_msgs::PointCloud2& msg,
                            std::vector<float>& intensity_out,
                            const std::string& data_set) {
    V3d_i result;
    intensity_out.clear();

    const auto* fx = detail::FindField(msg, "x");
    const auto* fy = detail::FindField(msg, "y");
    const auto* fz = detail::FindField(msg, "z");
    if (!fx || !fy || !fz) {
        mprtln(RED, "[online error] PointCloud2 missing x/y/z fields!");
        return result;
    }

    const auto* f_intensity = detail::FindField(msg, "intensity");
    if (!f_intensity) f_intensity = detail::FindField(msg, "i");

    // 优先 sem_label（推理节点发布的原始类别），其次 label（离线 bag 里就是原始类别）
    const auto* f_label = detail::FindField(msg, "sem_label");
    if (!f_label) f_label = detail::FindField(msg, "label");

    const IntensityNormParam norm = GetIntensityNormParam(data_set);

    const size_t num_points = static_cast<size_t>(msg.width) * msg.height;
    V3d points;
    std::vector<int> labels;
    points.reserve(num_points);
    labels.reserve(num_points);
    intensity_out.reserve(num_points);

    const uint8_t* data_ptr = msg.data.data();
    for (size_t row = 0; row < msg.height; ++row) {
        const uint8_t* row_ptr = data_ptr + row * msg.row_step;
        for (size_t col = 0; col < msg.width; ++col) {
            const uint8_t* ptr = row_ptr + col * msg.point_step;

            const float x = detail::ReadFloat(ptr, fx);
            const float y = detail::ReadFloat(ptr, fy);
            const float z = detail::ReadFloat(ptr, fz);
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) continue;
            points.emplace_back(x, y, z);

            float norm_i = 0.0f;
            if (f_intensity) {
                const float raw_i = detail::ReadFloat(ptr, f_intensity);
                norm_i = raw_i;
                if (norm.enable) {
                    const float range = norm.max_v - norm.min_v;
                    if (std::fabs(range) > 1e-6f) norm_i = (raw_i - norm.min_v) / range;
                    norm_i = std::min(1.0f, std::max(0.0f, norm_i));
                }
            }
            intensity_out.push_back(norm_i);

            const uint32_t raw_label = detail::ReadUInt(ptr, f_label) & 0x0000ffff;
            const auto it = label_map.find(static_cast<int>(raw_label));
            labels.push_back(it != label_map.end() ? it->second : 1);
        }
    }

    return std::make_pair(std::move(points), std::move(labels));
}

/**
 * @brief 线程安全的点云帧缓冲区：ROS 回调 push，前端主循环 wait-pop
 *        max_size>0 时若前端处理不过来，丢弃最旧的帧（在线模式必须丢帧而不是无限堆积）
 */
class OnlineCloudBuffer {
public:
    void SetMaxSize(size_t n) { max_size_ = n; }

    void Push(const sensor_msgs::PointCloud2ConstPtr& msg) {
        if (!msg) return;
        {
            std::lock_guard<std::mutex> lock(mtx_);
            if (max_size_ > 0 && queue_.size() >= max_size_) {
                queue_.pop_front();
                ++dropped_;
                if (dropped_ % 20 == 0) {
                    mprtln(YELLOW, "[online warn] cloud queue full (%zu), dropped %lu frames in total",
                           max_size_, static_cast<unsigned long>(dropped_));
                }
            }
            queue_.push_back(msg);
        }
        cv_.notify_one();
    }

    /// @return false 表示超时/已关闭，且队列为空
    bool WaitPop(sensor_msgs::PointCloud2ConstPtr& out, double timeout_sec) {
        std::unique_lock<std::mutex> lock(mtx_);
        if (!cv_.wait_for(lock, std::chrono::duration<double>(timeout_sec),
                          [this] { return shutdown_ || !queue_.empty(); })) {
            return false;
        }
        if (queue_.empty()) return false;
        out = queue_.front();
        queue_.pop_front();
        return true;
    }

    void Shutdown() {
        {
            std::lock_guard<std::mutex> lock(mtx_);
            shutdown_ = true;
        }
        cv_.notify_all();
    }

    size_t Size() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return queue_.size();
    }

    /// 因前端处理不过来而被丢弃的帧数
    size_t Dropped() const {
        std::lock_guard<std::mutex> lock(mtx_);
        return dropped_;
    }

private:
    mutable std::mutex mtx_;
    std::condition_variable cv_;
    std::deque<sensor_msgs::PointCloud2ConstPtr> queue_;
    size_t max_size_ = 30;
    size_t dropped_ = 0;
    bool shutdown_ = false;
};

}  // namespace dero_slam
