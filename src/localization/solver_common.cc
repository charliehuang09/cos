#include "localization/solver_common.h"

#include <cmath>
#include <numbers>

namespace localization {

auto Variance(int num_tags, double distance, double min_variance,
              double scalar) -> double {
  return distance * scalar / (num_tags * num_tags) + min_variance;
}

auto PoseOffField(frc::Pose3d pose) -> bool {
  constexpr double kFieldErrorMargin = 0.2;
  constexpr double kMaxRobotHeightError = 0.2;
  constexpr double kMaxRobotTilt = std::numbers::pi / 12.0;
  const double x = pose.X().value();
  const double y = pose.Y().value();
  const double z = pose.Z().value();
  const double roll = pose.Rotation().X().value();
  const double pitch = pose.Rotation().Y().value();
  const double yaw = pose.Rotation().Z().value();
  return !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z) ||
         !std::isfinite(roll) || !std::isfinite(pitch) ||
         !std::isfinite(yaw) || x < -kFieldErrorMargin ||
         x > 16.54 + kFieldErrorMargin || y < -kFieldErrorMargin ||
         y > 8.0 + kFieldErrorMargin ||
         std::abs(z) > kMaxRobotHeightError ||
         std::abs(roll) > kMaxRobotTilt ||
         std::abs(pitch) > kMaxRobotTilt;
}

}  // namespace localization
