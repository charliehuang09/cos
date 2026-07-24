#pragma once

#include <array>
#include <functional>
#include <mutex>
#include <string>
#include <utility>
#include <vector>
#include "camera/nvjpeg_decode_node.h"

#include <vpi/Types.h>
#include <vpi/algo/AprilTags.h>

#include <opencv2/core/types.hpp>

#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"

namespace apriltag {

class NvidiaTagDetections final : public control_loop::IMessage {
 public:
  struct tag_detection {
    int tag_id;
    std::array<cv::Point2d, 4> corners;
  };

  auto GetType() -> const std::type_info& override {
    return typeid(NvidiaTagDetections);
  }
  auto GetSize() -> size_t override {
    return sizeof(*this) + tag_detections.capacity() * sizeof(tag_detection);
  }

  NvidiaTagDetections(std::vector<tag_detection> tag_detections_)
      : tag_detections(std::move(tag_detections_)) {}

  std::vector<tag_detection> tag_detections;
};

class NvidiaApriltagDetectorNode final : public control_loop::INode {
 public:
  NvidiaApriltagDetectorNode(std::string_view input_channel,
                             std::string_view output_channel,
                             std::string_view config_path,
                             control_loop::ThreadPool& thread_pool);
  ~NvidiaApriltagDetectorNode() override;
  void WarmUp();
  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback) {
    callbacks_.emplace_back(callback);
  };
  void Callback(const control_loop::Context& context);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;

 private:
  auto Detect(const camera::DecodedJpegBuffer& buffer)
      -> std::vector<NvidiaTagDetections::tag_detection>;
  auto Detect(VPIImage image)
      -> std::vector<NvidiaTagDetections::tag_detection>;

 private:
  VPIImage input_ = nullptr;
  VPIContext context_ = nullptr;
  VPIPayload payload_ = nullptr;
  VPIArray detections_ = nullptr;
  VPIStream stream_ = nullptr;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::string input_channel_;
  std::string output_channel_;
  control_loop::ThreadPool& thread_pool_;
  std::mutex detect_mutex_;
  int width_;
  int height_;
};

}  // namespace apriltag
