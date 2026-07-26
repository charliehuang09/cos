#include "gamepiece/gamepiece_node.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

#include <frc/geometry/Rotation3d.h>
#include <frc/geometry/Transform3d.h>
#include <frc/geometry/Translation3d.h>
#include <units/angle.h>
#include <units/length.h>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace gamepiece {

GamepieceNode::GamepieceNode(std::unique_ptr<ObjectDetector> detector,
                             std::string_view input_path,
                             std::string_view output_path,
                             const nlohmann::json& intrinsics,
                             const nlohmann::json& extrinsics,
                             control_loop::ThreadPool& thread_pool)
    : detector_(std::move(detector)),
      input_path_(input_path),
      output_path_(output_path),
      thread_pool_(thread_pool),
      dependencies_({{input_path_, typeid(camera::DecodedJpegBuffer)}}),
      publications_({{output_path_, typeid(GamepieceDetections)}}),
      cam_cx_(intrinsics.at("cx").get<float>()),
      cam_cy_(intrinsics.at("cy").get<float>()),
      fx_(intrinsics.at("fx").get<float>()),
      fy_(intrinsics.at("fy").get<float>()),
      cam_pitch_(extrinsics.at("rotation_y").get<float>()),
      pinhole_height_(extrinsics.at("translation_z").get<float>()),
      cam_pose_(
          frc::Translation3d{
              units::meter_t{extrinsics.at("translation_x").get<float>()},
              units::meter_t{extrinsics.at("translation_y").get<float>()},
              units::meter_t{extrinsics.at("translation_z").get<float>()}},
          frc::Rotation3d{
              units::radian_t{extrinsics.at("rotation_x").get<float>()},
              units::radian_t{extrinsics.at("rotation_y").get<float>()},
              units::radian_t{extrinsics.at("rotation_z").get<float>()}}) {
  if (detector_ == nullptr) {
    throw std::invalid_argument("GamepieceNode requires an object detector");
  }
}

void GamepieceNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

auto GamepieceNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    const auto* frame =
        context->GetMessage<camera::DecodedJpegBuffer>(input_path_);
    if (frame == nullptr || frame->destination.channel[0] == nullptr) {
      return;
    }

    thread_pool_.Submit([this, context, frame] {
      auto detections =
          std::make_unique<GamepieceDetections>(RunDetection(*frame));
      LOG(INFO) << "Gamepiece detections channel=" << output_path_
                << " timestamp=" << frame->timestamp
                << " count=" << detections->detections.size();
      for (size_t i = 0; i < detections->detections.size(); ++i) {
        const gamepiece_detection_t& detection = detections->detections[i];
        const frc::Translation3d& translation = detection.pose.Translation();
        const frc::Rotation3d& rotation = detection.pose.Rotation();
        LOG(INFO) << "Gamepiece detection channel=" << output_path_
                  << " timestamp=" << frame->timestamp << " index=" << i
                  << " class_id=" << detection.class_id
                  << " confidence=" << detection.confidence
                  << " tracker_id=" << detection.tracker_id
                  << " pose(x=" << translation.X().value()
                  << " y=" << translation.Y().value()
                  << " z=" << translation.Z().value()
                  << " roll=" << rotation.X().value()
                  << " pitch=" << rotation.Y().value()
                  << " yaw=" << rotation.Z().value() << ")";
      }
      context->SetMessage(output_path_, std::move(detections));
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    });
  };
}

auto GamepieceNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto GamepieceNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

auto GamepieceNode::RunDetection(const camera::DecodedJpegBuffer& frame)
    -> GamepieceDetections {
  CHECK_EQ(frame.output_format, NVJPEG_OUTPUT_Y);
  const cv::cuda::GpuMat image(frame.height, frame.width, CV_8UC1,
                               frame.destination.channel[0], frame.stride);

  std::lock_guard lock(detection_mutex_);
  const std::vector<LabeledBoundingBox> boxes = detector_->Detect(image);
  GamepieceDetections result;
  result.detections.reserve(boxes.size());
  for (const LabeledBoundingBox& box : boxes) {
    result.detections.push_back(gamepiece_detection_t{
        .pose = ComputePose(box.bounds),
        .tracker_id = -1,
        .class_id = box.class_id,
        .confidence = box.confidence,
    });
  }
  return result;
}

auto GamepieceNode::ComputePose(const cv::Rect& bbox) const -> frc::Pose3d {
  const float center_y = bbox.y + bbox.height / 2.0F;
  const float center_x = bbox.x + bbox.width / 2.0F;
  const float cam_relative_pitch = std::atan2(center_y - cam_cy_, fy_);
  const float phi = cam_relative_pitch + cam_pitch_;
  const float distance = pinhole_height_ / std::sin(phi);
  const float cam_relative_yaw = std::atan2(center_x - cam_cx_, fx_);

  const frc::Transform3d target_pose_cam_relative{
      frc::Translation3d{
          units::meter_t{distance * std::cos(cam_relative_pitch) *
                         std::cos(cam_relative_yaw)},
          units::meter_t{distance * std::cos(cam_relative_pitch) *
                         std::sin(cam_relative_yaw)},
          units::meter_t{distance * -std::sin(cam_relative_pitch)}},
      frc::Rotation3d{units::radian_t{0.0},
                      units::radian_t{cam_relative_pitch},
                      units::radian_t{-cam_relative_yaw}}};

  return cam_pose_.TransformBy(target_pose_cam_relative);
}

}  // namespace gamepiece
