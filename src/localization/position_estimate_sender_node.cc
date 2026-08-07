
#include "localization/position_estimate_sender_node.h"
#include "absl/log/log.h"
#include "localization/position.h"

namespace localization {

PositionEstimateSenderNode::PositionEstimateSenderNode(
    std::string_view input_channel, std::string_view networktables_channel,
    const nt::NetworkTableInstance& instance)
    : input_channel_(input_channel) {
  dependencies_.emplace_back(input_channel, typeid(PositionEstimateMessage));
  std::shared_ptr<nt::NetworkTable> table =
      instance.GetTable(networktables_channel);

  nt::StructTopic<frc::Pose2d> pose2d_topic =
      table->GetStructTopic<frc::Pose2d>("Pose2d");
  nt::StructTopic<frc::Pose3d> pose3d_topic =
      table->GetStructTopic<frc::Pose3d>("Pose3d");
  pose2d_publisher_ = pose2d_topic.Publish();
  pose3d_publisher_ = pose3d_topic.Publish();

  nt::DoubleArrayTopic pose_topic = table->GetDoubleArrayTopic("TagEstimation");
  pose_publisher_ = pose_topic.Publish(
      {.pollStorage = 200, .sendAll = true, .keepDuplicates = true});
}

auto PositionEstimateSenderNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto PositionEstimateSenderNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void PositionEstimateSenderNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto PositionEstimateSenderNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto position_estimate =
        context->GetMessage<PositionEstimateMessage>(input_channel_);
    if (position_estimate == nullptr) {
      return;
    }
    std::array<double, 4> pose{
        position_estimate->pose.X().value(),
        position_estimate->pose.Y().value(),
        position_estimate->pose.Z().value(),
        position_estimate->variance,
    };
    pose_publisher_.Set(pose);
    pose3d_publisher_.Set(position_estimate->pose);
    pose2d_publisher_.Set(position_estimate->pose.ToPose2d());
    if (log_estimate_) {
      LOG(INFO) << *position_estimate;
    }
  };
}

void PositionEstimateSenderNode::SetLogEstimates(bool log_estimate) {
  log_estimate_ = log_estimate;
}

}  // namespace localization
