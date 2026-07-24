#pragma once

#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <typeinfo>
#include <utility>
#include <vector>

#include <opencv2/core/types.hpp>
#include <wpi/apriltag/AprilTagFieldLayout.hpp>
#include <wpi/math/geometry/Pose3d.hpp>

#include "apriltag/nvidia_apriltag_detector_node.h"
#include "localization/position.h"

namespace localization {

using tag_detection_t = apriltag::NvidiaTagDetections::tag_detection;

struct AmbiguousEstimate {
  position_estimate_t pos1;
  std::optional<position_estimate_t> pos2;
};

using ambiguous_estimate_t = AmbiguousEstimate;

class AmbiguousEstimateMessage final : public control_loop::IMessage {
 public:
  explicit AmbiguousEstimateMessage(std::vector<AmbiguousEstimate> estimates)
      : estimates(std::move(estimates)) {}

  auto GetType() -> const std::type_info& override {
    return typeid(AmbiguousEstimateMessage);
  }
  auto GetSize() -> std::size_t override {
    return sizeof(*this) + estimates.capacity() * sizeof(AmbiguousEstimate);
  }

  std::vector<AmbiguousEstimate> estimates;
};

struct CameraSolverConfig {
  std::string name;
  std::string intrinsics_path;
  std::string extrinsics_path;
};

using camera_constant_t = CameraSolverConfig;

inline auto DetectionBatchChannel(std::string_view camera_name) -> std::string {
  return "localization/" + std::string(camera_name) + "_detection_batch";
}

constexpr double kTagSize = 0.1651;
constexpr double kMinTagAreaPixels = 100.0;
constexpr double kMaxTagDistance = 5.0;

inline const std::vector<cv::Point3d> kApriltagCorners = {
    {-kTagSize / 2.0, kTagSize / 2.0, 0.0},
    {kTagSize / 2.0, kTagSize / 2.0, 0.0},
    {kTagSize / 2.0, -kTagSize / 2.0, 0.0},
    {-kTagSize / 2.0, -kTagSize / 2.0, 0.0}};

inline const wpi::apriltag::AprilTagFieldLayout kApriltagLayout =
    wpi::apriltag::AprilTagFieldLayout::LoadField(
        wpi::apriltag::AprilTagField::k2026RebuiltAndyMark);

auto Variance(int num_tags, double distance, double min_variance,
              double scalar) -> double;
auto PoseOffField(wpi::math::Pose3d pose) -> bool;

}  // namespace localization
