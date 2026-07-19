#include "localization/solver_common.h"


namespace localization {

auto Variance(int num_tags, double distance, double min_variance,
              double scalar) -> double {
  return distance * scalar / (num_tags * num_tags) + min_variance;
}

auto PoseOffField(wpi::math::Pose3d pose) -> bool {
  constexpr double kErrorMargin = 0.2;
  return pose.X().value() < -kErrorMargin ||
         pose.X().value() > 16.54 + kErrorMargin ||
         pose.Y().value() < -kErrorMargin ||
         pose.Y().value() > 8.0 + kErrorMargin;
}

}  // namespace localization
