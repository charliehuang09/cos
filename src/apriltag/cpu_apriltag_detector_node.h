#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <wpi/apriltag/AprilTagDetector.hpp>

#include "apriltag/tag_detections.h"
#include "camera/cpu_decode_node.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"

namespace apriltag {

class CpuApriltagDetectorNode final : public control_loop::INode {
 public:
  CpuApriltagDetectorNode(std::string_view input_channel,
                          std::string_view output_channel,
                          std::string_view config_path,
                          control_loop::ThreadPool& thread_pool);

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override {
    callbacks_.emplace_back(callback);
  }
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  auto Detect(const camera::DecodedImageBuffer& image)
      -> std::vector<TagDetections::tag_detection>;

  wpi::apriltag::AprilTagDetector detector_;
  std::string input_channel_;
  std::string output_channel_;
  control_loop::ThreadPool& thread_pool_;
  std::mutex detect_mutex_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

}  // namespace apriltag
