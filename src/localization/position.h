#pragma once

#include <ostream>
#include <typeinfo>
#include <utility>
#include <vector>

#include <wpi/math/geometry/Pose3d.hpp>

#include "control_loop/message.h"

namespace localization {

struct PositionEstimate final : public control_loop::IMessage {
  std::vector<int> tag_ids;
  std::vector<int> rejected_tag_ids;
  wpi::math::Pose3d pose;
  double variance = 0.0;
  int num_tags = 0;
  double avg_tag_dist = 0.0;
  bool invalid = false;
  double loss = 0.0;

  PositionEstimate() = default;
  PositionEstimate(const PositionEstimate& other)
      : control_loop::IMessage(),
        tag_ids(other.tag_ids),
        rejected_tag_ids(other.rejected_tag_ids),
        pose(other.pose),
        variance(other.variance),
        num_tags(other.num_tags),
        avg_tag_dist(other.avg_tag_dist),
        invalid(other.invalid),
        loss(other.loss) {}
  PositionEstimate(PositionEstimate&& other) noexcept
      : control_loop::IMessage(),
        tag_ids(std::move(other.tag_ids)),
        rejected_tag_ids(std::move(other.rejected_tag_ids)),
        pose(std::move(other.pose)),
        variance(other.variance),
        num_tags(other.num_tags),
        avg_tag_dist(other.avg_tag_dist),
        invalid(other.invalid),
        loss(other.loss) {}
  auto operator=(const PositionEstimate& other) -> PositionEstimate& {
    if (this == &other) {
      return *this;
    }
    tag_ids = other.tag_ids;
    rejected_tag_ids = other.rejected_tag_ids;
    pose = other.pose;
    variance = other.variance;
    num_tags = other.num_tags;
    avg_tag_dist = other.avg_tag_dist;
    invalid = other.invalid;
    loss = other.loss;
    return *this;
  }
  auto operator=(PositionEstimate&& other) noexcept -> PositionEstimate& {
    if (this == &other) {
      return *this;
    }
    tag_ids = std::move(other.tag_ids);
    rejected_tag_ids = std::move(other.rejected_tag_ids);
    pose = std::move(other.pose);
    variance = other.variance;
    num_tags = other.num_tags;
    avg_tag_dist = other.avg_tag_dist;
    invalid = other.invalid;
    loss = other.loss;
    return *this;
  }

  auto GetType() -> const std::type_info& override {
    return typeid(PositionEstimate);
  }

  friend auto operator<<(std::ostream& os,
                         const PositionEstimate& estimate) -> std::ostream& {
    const auto& translation = estimate.pose.Translation();
    const auto& rotation = estimate.pose.Rotation();
    os << "pose(x=" << translation.X().value()
       << " y=" << translation.Y().value()
       << " z=" << translation.Z().value()
       << " roll=" << rotation.X().value()
       << " pitch=" << rotation.Y().value()
       << " yaw=" << rotation.Z().value() << ")"
       << " variance=" << estimate.variance
       << " num_tags=" << estimate.num_tags;
    return os;
  }
};

using position_estimate_t = PositionEstimate;

}  // namespace localization
