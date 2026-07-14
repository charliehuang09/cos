#pragma once

#include <array>
#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "camera/nvjpeg_decode_node.h"

#include <vpi/Types.h>
#include <vpi/algo/AprilTags.h>

#include <opencv2/core/types.hpp>

#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"

namespace apriltag {

class NvidiaTagDetections {
 public:
  struct tag_detection {
    int tag_id;
    std::array<cv::Point2d, 4> corners;
  };
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
      const std::function<void(const std::shared_ptr<NvidiaTagDetections>&)>&
          callback);
  void Callback(const control_loop::Context& context);
  void Detect(const camera::DecodedJpegBuffer& buffer);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;

 private:
  void Detect(VPIImage image);

 private:
  VPIImage input_ = nullptr;
  VPIContext context_ = nullptr;
  VPIPayload payload_ = nullptr;
  VPIArray detections_ = nullptr;
  VPIStream stream_ = nullptr;
  std::vector<std::function<void(const std::shared_ptr<NvidiaTagDetections>&)>>
      callbacks_;
  std::string input_channel_;
  std::string output_channel_;
  control_loop::ThreadPool& thread_pool_;
  std::mutex detect_mutex_;
  int width_;
  int height_;
};

}  // namespace apriltag
