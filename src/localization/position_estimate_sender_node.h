#pragma once

#include <frc/geometry/Pose2d.h>
#include <frc/geometry/Pose3d.h>
#include <networktables/BooleanArrayTopic.h>
#include <networktables/BooleanTopic.h>
#include <networktables/DoubleArrayTopic.h>
#include <networktables/DoubleTopic.h>
#include <networktables/IntegerTopic.h>
#include <networktables/NetworkTable.h>
#include <networktables/NetworkTableInstance.h>
#include <networktables/StructArrayTopic.h>
#include <networktables/StructTopic.h>
#include <ntcore/networktables/NetworkTableInstance.h>
#include "control_loop/message.h"
#include "control_loop/node.h"

namespace localization {

class PositionEstimateSenderNode final : public control_loop::INode {
 public:
  PositionEstimateSenderNode(std::string_view input_channel,
                             std::string_view networktables_channel,
                             const nt::NetworkTableInstance& instance);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  void SetLogEstimates(bool log_estimate);

 private:
  nt::StructPublisher<frc::Pose2d> pose2d_publisher_;
  nt::StructPublisher<frc::Pose3d> pose3d_publisher_;
  nt::DoubleArrayPublisher pose_publisher_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::string input_channel_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  bool log_estimate_ = false;
};
}  // namespace localization
