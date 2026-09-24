#include "logging/estimate_log.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <frc/geometry/struct/Pose3dStruct.h>
#include <nlohmann/json.hpp>
#include <wpi/struct/Struct.h>

#include "localization/position.h"
#include "localization/solver_common.h"
#include "logging/wpilog_writer.h"

namespace logging {
namespace {

auto EncodePose(const frc::Pose3d& pose) -> nlohmann::json {
  const auto quaternion = pose.Rotation().GetQuaternion();
  return {{"x", pose.X().value()},
          {"y", pose.Y().value()},
          {"z", pose.Z().value()},
          {"qw", quaternion.W()},
          {"qx", quaternion.X()},
          {"qy", quaternion.Y()},
          {"qz", quaternion.Z()}};
}

auto EncodeEstimate(const localization::SolverEstimate& estimate)
    -> nlohmann::json {
  return {{"pose", EncodePose(estimate.pose)},
          {"tag_ids", estimate.tag_ids},
          {"distances", estimate.distances},
          {"variance", estimate.variance},
          {"distance", estimate.distance}};
}

auto PackPoses(const std::vector<frc::Pose3d>& poses) -> std::vector<uint8_t> {
  constexpr size_t kPoseSize = wpi::GetStructSize<frc::Pose3d>();
  std::vector<uint8_t> bytes(poses.size() * kPoseSize);
  for (size_t index = 0; index < poses.size(); ++index) {
    wpi::PackStruct(std::span<uint8_t>(bytes).subspan(index * kPoseSize, kPoseSize),
                    poses[index]);
  }
  return bytes;
}

}  // namespace

template <>
std::vector<int> StartLog<localization::PositionEstimateMessage>(
    WPILogWriter& writer, std::string_view channel) {
  writer.AddPose3dSchema();
  const std::string name(channel);
  return {writer.StartRaw(channel, wpi::GetStructTypeString<frc::Pose3d>()),
          writer.StartIntegerArray(name + "/tag_ids"),
          writer.StartDoubleArray(name + "/distances"), writer.StartDouble(name + "/num_tags")};
}

template <>
void AppendLog<localization::PositionEstimateMessage>(
    WPILogWriter& writer, std::span<const int> entries,
    const localization::PositionEstimateMessage& message, int64_t timestamp) {
  std::array<uint8_t, wpi::GetStructSize<frc::Pose3d>()> pose_bytes;
  wpi::PackStruct(pose_bytes, message.pose);
  writer.AppendRaw(entries[0], pose_bytes, timestamp);
  const std::vector<int64_t> tag_ids(message.tag_ids.begin(),
                                     message.tag_ids.end());
  writer.AppendIntegerArray(entries[1], tag_ids, timestamp);
  writer.AppendDoubleArray(entries[2], message.distances, timestamp);
  writer.AppendDouble(entries[3], message.tag_ids.size(), timestamp);
}

template <>
std::vector<int> StartLog<localization::VarianceMessage>(
    WPILogWriter& writer, std::string_view channel) {
  return {writer.StartDouble(channel)};
}

template <>
void AppendLog<localization::VarianceMessage>(
    WPILogWriter& writer, std::span<const int> entries,
    const localization::VarianceMessage& message, int64_t timestamp) {
  writer.AppendDouble(entries[0], message.value, timestamp);
}

template <>
std::vector<int> StartLog<localization::AmbiguousEstimateMessage>(
    WPILogWriter& writer, std::string_view channel) {
  writer.AddPose3dSchema();
  const std::string name(channel);
  const std::string pose_array_type =
      std::string(wpi::GetStructTypeString<frc::Pose3d>()) + "[]";
  return {writer.StartString(channel),
          writer.StartRaw(name + "/pos1", pose_array_type),
          writer.StartRaw(name + "/pos2", pose_array_type),
          writer.StartIntegerArray(name + "/pos2_indices"),
          writer.StartDoubleArray(name + "/pos1_variance"),
          writer.StartDoubleArray(name + "/pos2_variance"),
          writer.StartDoubleArray(name + "/pos1_distance"),
          writer.StartDoubleArray(name + "/pos2_distance")};
}

template <>
void AppendLog<localization::AmbiguousEstimateMessage>(
    WPILogWriter& writer, std::span<const int> entries,
    const localization::AmbiguousEstimateMessage& message, int64_t timestamp) {
  nlohmann::json values = nlohmann::json::array();
  std::vector<frc::Pose3d> pos1_poses;
  std::vector<frc::Pose3d> pos2_poses;
  std::vector<int64_t> pos2_indices;
  std::vector<double> pos1_variances;
  std::vector<double> pos2_variances;
  std::vector<double> pos1_distances;
  std::vector<double> pos2_distances;
  for (size_t index = 0; index < message.estimates.size(); ++index) {
    const auto& estimate = message.estimates[index];
    nlohmann::json value = {{"pos1", EncodeEstimate(estimate.pos1)}};
    pos1_poses.push_back(estimate.pos1.pose);
    pos1_variances.push_back(estimate.pos1.variance);
    pos1_distances.push_back(estimate.pos1.distance);
    if (estimate.pos2.has_value()) {
      value["pos2"] = EncodeEstimate(*estimate.pos2);
      pos2_poses.push_back(estimate.pos2->pose);
      pos2_indices.push_back(static_cast<int64_t>(index));
      pos2_variances.push_back(estimate.pos2->variance);
      pos2_distances.push_back(estimate.pos2->distance);
    } else {
      value["pos2"] = nullptr;
    }
    values.push_back(std::move(value));
  }
  writer.AppendString(entries[0], values.dump(), timestamp);
  writer.AppendRaw(entries[1], PackPoses(pos1_poses), timestamp);
  writer.AppendRaw(entries[2], PackPoses(pos2_poses), timestamp);
  writer.AppendIntegerArray(entries[3], pos2_indices, timestamp);
  writer.AppendDoubleArray(entries[4], pos1_variances, timestamp);
  writer.AppendDoubleArray(entries[5], pos2_variances, timestamp);
  writer.AppendDoubleArray(entries[6], pos1_distances, timestamp);
  writer.AppendDoubleArray(entries[7], pos2_distances, timestamp);
}

}  // namespace logging
