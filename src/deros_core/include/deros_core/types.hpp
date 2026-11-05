#pragma once

/// @file types.hpp
/// Point cloud containers used across the public API.
///
/// The application may define the same aliases (it does, through
/// deros_core/labels.hpp); typedefs are interchangeable, so there is no clash.

#include <utility>
#include <vector>

#include <Eigen/Core>

typedef std::vector<Eigen::Vector3d> V3d;
typedef std::pair<V3d, std::vector<int>> V3d_i;

typedef std::vector<Eigen::Matrix<double, 6, 1>> V6d;
typedef std::pair<V6d, std::vector<int>> V6d_i;

typedef std::vector<Eigen::Vector4d> V4d;
