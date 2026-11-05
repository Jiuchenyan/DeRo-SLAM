#pragma once

/// @file ops.hpp
/// Free functions of the library that operate on plain point clouds.

#include "deros_core/types.hpp"

namespace deros_api {

/// Voxel-grid subsampling of a labelled cloud: one point per voxel, the mean of
/// the points falling into it, the label of the first point.
V3d_i DownsampleMean(const V3d_i &frame, double voxel_size);

}   // namespace deros_api
