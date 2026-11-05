#pragma once

#include <iostream>
#include <algorithm>
#include <vector>
#include <Eigen/Core>
#include <chrono>
#include <unordered_map>

typedef std::vector<Eigen::Vector3d> V3d;
typedef std::pair<V3d,std::vector<int>> V3d_i;

typedef std::vector<Eigen::Matrix<double, 6, 1>> V6d;
typedef std::pair<V6d,std::vector<int>> V6d_i;

typedef std::vector<Eigen::Vector4d> V4d;


// Hash function for Eigen::Vector3i
struct VoxelHash{
            size_t operator()(const Eigen::Vector3i &voxel) const {
                const uint32_t *vec = reinterpret_cast<const uint32_t *>(voxel.data());
                return ((1 << 20) - 1) & (vec[0] * 73856093 ^ vec[1] * 19349663 ^ vec[2] * 83492791);
                }
}; 
using Voxel = Eigen::Vector3i;

/*  true*/
// Label mapping, Note: mulran mapping all vehicle label -> 0
inline std::unordered_map<int, int> label_map={
  {0, 0},     // "unlabeled"
  {1, 0},     // "outlier" mapped to "unlabeled" --------------------------mapped
  {10, 1},     //! "car"  1
  {11, 2},     // "bicycle"
  {13, 5},     // 5 "bus" mapped to "other-vehicle" --------------------------mapped
  {15, 3},     // "motorcycle"
  {16, 5},     // 5"on-rails" mapped to "other-vehicle" ---------------------mapped
  {18, 4},     // 4"truck"
  {20, 5},     // 5"other-vehicle"
  {30, 6},     // "person"
  {31, 7},     // "bicyclist"
  {32, 8},     // "motorcyclist"
  {40, 9},     // 9"road"
  {44, 10},    // 10"parking"
  {48, 11},    // 11"sidewalk"
  {49, 12},    // 12"other-ground"
  {50, 13},    // 13"building"
  {51, 14},    // 14"fence"
  {52, 0},     // "other-structure" mapped to "unlabeled" ------------------mapped
  {60, 9},     // 9"lane-marking" to "road" ---------------------------------mapped
  {70, 15},    // 15"vegetation"
  {71, 16},    // 16"trunk"
  {72, 17},    // 17"terrain"
  {80, 18},    // 18"pole"
  {81, 19},    // 19"traffic-sign"
  {99, 0},     // "other-object" to "unlabeled" ----------------------------mapped
  {252, 20},    // "moving-car"
  {253, 21},    // "moving-bicyclist"
  {254, 22},    // "moving-person"
  {255, 23},    // "moving-motorcyclist"
  {256, 24},    // "moving-on-rails" mapped to "moving-other-vehicle" ------mapped
  {257, 24},    // "moving-bus" mapped to "moving-other-vehicle" -----------mapped
  {258, 25},    // "moving-truck"
  {259, 24}    // "moving-other-vehicle"

};




// inline std::unordered_map<int, int> label_map={
//   {0, 0},     // Invalue
//   {1, 1},     // car
//   {2, 2},     // bycicle
//   {6, 254},     // person
//   {9, 9},     // road
//   {13, 13},   // building
//   {14, 14},   // fence
//   {15, 15},   // vegetation
//   {16, 16},   // trunk
//   {18, 18},   // pole
//   {255, 254}  // outlier
// };



namespace SemanticLabel {
constexpr int Invalue         = 0;

constexpr int car             = 1;
constexpr int bicycle         = 2;
constexpr int motorcycle      = 3;
constexpr int truck           = 4;
constexpr int bus             = 5;
constexpr int person          = 6;
constexpr int bicyclist       = 7;
constexpr int motorcyclist    = 8;

constexpr int fence           = 14;
constexpr int vegetation      = 15;
constexpr int trunk           = 16;
constexpr int pole            = 18;
constexpr int trafficsign     = 19;

constexpr int road            = 9;
constexpr int parking         = 10;
constexpr int sidewalk        = 11;
constexpr int otherground     = 12;
constexpr int building        = 13;
constexpr int terrain         = 17;

constexpr bool BigLevel1(int label){
    return label == trafficsign;
}

constexpr bool BigLevel2(int label){
    return label == pole;
}

constexpr bool BigLevel3(int label){
    return label == trunk;
}

constexpr bool BigLevel4(int label){
    return label == car;
}

constexpr bool BigLevel5(int label){
    return label == truck;
}

constexpr bool BigLevel6(int label){
    return label == bus || label == fence;
}

constexpr bool BigLevel7(int label){
    return label == vegetation;
}

constexpr bool IsForegroundPoint(int label) {
    return label == car || label == truck || label == bus ||
            label == fence || label == vegetation || label == trunk ||
            label == pole || label == trafficsign;
}

constexpr bool IsBackgroundPoint(int label) {
    return label == road || label == parking || label == sidewalk || label == terrain || label == otherground || 
            label == building || label == fence;
}

constexpr bool IsGroundPoint(int label) {
    return label == road || label == parking || label == sidewalk ||
            label == otherground|| label == terrain;
}

constexpr bool IsCarPoint(int label) {
    return label == car || label == truck || label == bus;
}

constexpr bool IsDynamic(int label){
    return  label == car || label == truck || label == bus||
            label == bicycle || label == motorcycle || 
            label == person|| label == bicyclist|| label == motorcyclist;
}

constexpr bool IsOutlinePoint(int label) {
    return label == building || label == fence;
}

constexpr bool IsBevNeedForMapping(int label) {
    return  label == building || label == fence || 
            label == pole || label == trunk ||
            label == vegetation;
}

constexpr bool IsSemanticNeedForMapping(int label) {
    return  label == building || label == fence || 
            label == pole || label == trunk;
}

constexpr bool IsBackgroundNeedForMapping(int label) {
    return label == building || label == fence;
}

constexpr bool IsForegroundNeedForMapping(int label) {
    return label == pole || label == trunk || label == trafficsign;
}

constexpr bool IsBackgroundNeedForLoop(int label) {
    return  label == road || label == parking || label == sidewalk || label == otherground || 
            label == terrain ||
            label == vegetation || 
            label == fence || 
            label == building;
}

constexpr double GroundWeight(int label) {
    if (label == road)        return 0.5;
    if (label == sidewalk)    return 0.95;
    if (label == parking)     return 0.9;
    if (label == otherground) return 0.9;
    if (label == terrain)     return 0.85;
    return 0.0;
}
}

// Color map for visualization
inline std::unordered_map<int,Eigen::Vector3i> color_map={
			{0,{0,0,0}},
			{0,{255,0,0}},
			{1,{100,150,245}},
			{2,{100,230,245}},
			{5,{100,80,250}},
			{3,{30,60,150}},
			{5,{0,0,255}},
			{4,{80,30,180}},
			{5,{0,0,255}},
			{6,{255,30,30}},
			{7,{255,40,200}},
			{8,{150,30,90}},
			{9,{255,0,255}},
			{10,{255,150,255}},
			{11,{75,0,75}},
			{12,{175,0,75}},
			{13,{255,200,0}},
			{14,{255,120,50}},
			{0,{255,150,0}},
			{9,{150,255,170}},
			{15,{0,175,0}},
			{16,{135,60,0}},
			{17,{150,240,80}},
			{18,{255,240,150}},
			{19,{255,0,0}},
			{0,{50,255,255}},
			{20,{100,150,245}},
			{21,{255,40,200}},
			{22,{255,30,30}},
			{23,{150,30,90}},
			{24,{0,0,255}},
			{24,{100,80,250}},
			{25,{80,30,180}},
			{24,{255,0,0}},

            {99, {0, 0, 0}},

            {100, {255, 255, 0}},
            {101, {255, 0, 0}},
            {102, {0, 255, 0}},
            {103, {0, 0, 255}}

};

