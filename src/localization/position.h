#pragma once

#include <ostream>
#include <typeinfo>
#include <utility>
#include <vector>

#include <frc/geometry/Pose3d.h>

#include "control_loop/message.h"

namespace localization {

struct PositionEstimateMessage final : public control_loop::IMessage {
  std::vector<int> tag_ids;
  frc::Pose3d pose;
  std::vector<double> distances;

  PositionEstimateMessage() = default;
  PositionEstimateMessage(const PositionEstimateMessage& other)
      : control_loop::IMessage(),
        tag_ids(other.tag_ids),
        pose(other.pose),
        distances(other.distances) {}
  PositionEstimateMessage(PositionEstimateMessage&& other) noexcept
      : control_loop::IMessage(),
        tag_ids(std::move(other.tag_ids)),
        pose(other.pose),
        distances(std::move(other.distances)) {}
  auto operator=(const PositionEstimateMessage& other)
      -> PositionEstimateMessage& {
    if (this == &other) {
      return *this;
    }
    tag_ids = other.tag_ids;
    pose = other.pose;
    distances = other.distances;
    return *this;
  }
  auto operator=(PositionEstimateMessage&& other) noexcept
      -> PositionEstimateMessage& {
    if (this == &other) {
      return *this;
    }
    tag_ids = std::move(other.tag_ids);
    pose = other.pose;
    distances = std::move(other.distances);
    return *this;
  }

  auto GetType() -> const std::type_info& override {
    return typeid(PositionEstimateMessage);
  }
  auto GetSize() -> std::size_t override {
    return sizeof(*this) + tag_ids.capacity() * sizeof(int) +
           distances.capacity() * sizeof(double);
  }

  friend auto operator<<(std::ostream& os,
                         const PositionEstimateMessage& estimate)
      -> std::ostream& {
    const auto& translation = estimate.pose.Translation();
    const auto& rotation = estimate.pose.Rotation();
    os << "pose(x=" << translation.X().value()
       << " y=" << translation.Y().value() << " z=" << translation.Z().value()
       << " roll=" << rotation.X().value() << " pitch=" << rotation.Y().value()
       << " yaw=" << rotation.Z().value() << ")"
       << " distances=[";
    for (std::size_t i = 0; i < estimate.distances.size(); ++i) {
      if (i != 0) {
        os << ',';
      }
      os << estimate.distances[i];
    }
    os << "] tag_ids=[";
    for (std::size_t i = 0; i < estimate.tag_ids.size(); ++i) {
      if (i != 0) {
        os << ',';
      }
      os << estimate.tag_ids[i];
    }
    return os << "]";
  }
};

struct VarianceMessage final : public control_loop::IMessage {
  explicit VarianceMessage(double value) : value(value) {}

  auto GetType() -> const std::type_info& override {
    return typeid(VarianceMessage);
  }
  auto GetSize() -> std::size_t override { return sizeof(*this); }

  double value;
};

using position_estimate_t = PositionEstimateMessage;

}  // namespace localization
