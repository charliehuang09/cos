#pragma once

#include <functional>
#include <mutex>
#include <string>
#include <vector>
#include "apriltag/tag_detections.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "camera/nvjpeg_decode_node.h"

#include <vpi/Types.h>
#include <vpi/algo/AprilTags.h>
#include <opencv2/core/mat.hpp>

#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"

namespace apriltag {

class NvidiaApriltagDetectorNode final : public control_loop::INode {
 public:
  NvidiaApriltagDetectorNode(std::string_view input_channel,
                             std::string_view output_channel,
                             std::string_view config_path,
                             control_loop::ThreadPool& thread_pool);
  ~NvidiaApriltagDetectorNode() override;
  void WarmUp();
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override {
    callbacks_.emplace_back(callback);
  };
  void Callback(const control_loop::Context& context);  // TODO Private?
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  auto Detect(const camera::DecodedJpegBuffer& buffer)
      -> std::vector<TagDetections::tag_detection>;
  auto Detect(const camera::DecodedJpegFdBuffer& buffer)
      -> std::vector<TagDetections::tag_detection>;
  auto DetectGray(const unsigned char* data, int width, int height,
                  size_t stride)
      -> std::vector<TagDetections::tag_detection>;
  auto Detect(VPIImage image)
      -> std::vector<TagDetections::tag_detection>;

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
  cv::Mat camera_matrix_;
  cv::Mat distortion_coefficients_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

}  // namespace apriltag
