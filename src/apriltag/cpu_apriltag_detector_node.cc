#include "apriltag/cpu_apriltag_detector_node.h"

#include <fstream>
#include <memory>

#include <nlohmann/json.hpp>

#include "absl/log/check.h"

namespace apriltag {

CpuApriltagDetectorNode::CpuApriltagDetectorNode(
    std::string_view input_channel, std::string_view output_channel,
    std::string_view config_path, control_loop::ThreadPool& thread_pool)
    : input_channel_(input_channel),
      output_channel_(output_channel),
      thread_pool_(thread_pool),
      dependencies_({{input_channel_, typeid(camera::DecodedImageBuffer)}}),
      publications_({{output_channel_, typeid(TagDetections)}}) {
  std::ifstream config_file{std::string(config_path)};
  CHECK(config_file.is_open()) << "Failed to open config: " << config_path;
  const nlohmann::json config = nlohmann::json::parse(config_file);
  CHECK_GT(config.at("width").get<int>(), 0);
  CHECK_GT(config.at("height").get<int>(), 0);

  frc::AprilTagDetector::Config detector_config;
  detector_config.numThreads = 1;
  detector_config.quadDecimate = 2.0F;
  detector_.SetConfig(detector_config);
  CHECK(detector_.AddFamily("tag36h11"));
}

auto CpuApriltagDetectorNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    const auto* image =
        context->GetMessage<camera::DecodedImageBuffer>(input_channel_);
    if (image == nullptr) {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
      return;
    }

    thread_pool_.Submit(
        [this, context, image] -> void {
          auto detections = std::make_unique<TagDetections>(Detect(*image));
          context->SetMessage(output_channel_, std::move(detections));
          for (const auto& callback : callbacks_) {
            callback(context);
          }
        },
        context->id);
  };
}

auto CpuApriltagDetectorNode::Detect(const camera::DecodedImageBuffer& image)
    -> std::vector<TagDetections::tag_detection> {
  CHECK_EQ(image.data.size(), image.stride * static_cast<size_t>(image.height));
  CHECK_EQ(image.stride, static_cast<size_t>(image.width));
  std::lock_guard lock(detect_mutex_);
  auto* pixels = const_cast<uint8_t*>(image.data.data());
  auto results = detector_.Detect(image.width, image.height,
                                  static_cast<int>(image.stride), pixels);

  std::vector<TagDetections::tag_detection> detections;
  detections.reserve(results.size());
  for (const auto* result : results) {
    TagDetections::tag_detection detection;
    detection.tag_id = result->GetId();
    for (int corner = 0; corner < 4; ++corner) {
      const auto point = result->GetCorner(corner);
      detection.corners[corner] = cv::Point2d{point.x, point.y};
    }
    detections.push_back(detection);
  }
  return detections;
}

auto CpuApriltagDetectorNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto CpuApriltagDetectorNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace apriltag
