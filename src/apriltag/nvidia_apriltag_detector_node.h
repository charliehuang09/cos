#pragma once

#include <array>
#include <functional>
#include <string>
#include <vector>
#include "camera/nvjpeg_decode_node.h"

#include <vpi/Types.h>
#include <vpi/algo/AprilTags.h>

#include <opencv2/core/types.hpp>

namespace apriltag {

class NvidiaTagDetections {
 public:
  struct tag_detection {
    int tag_id;
    std::array<cv::Point2d, 4> corners;
  };
  std::vector<tag_detection> tag_detections;
};

class NvidiaApriltagDetectorNode {
 public:
  NvidiaApriltagDetectorNode(int width, int height, std::string& config_path);
  ~NvidiaApriltagDetectorNode();
  void RegisterCallback(
      const std::function<void(const std::shared_ptr<NvidiaTagDetections>&)>&
          callback);
  void Detect(const std::shared_ptr<camera::DecodedJpegBuffer>& buffer);

 private:
  void Detect(VPIImage image);

 private:
  VPIImage input_ = nullptr;
  VPIPayload payload_ = nullptr;
  VPIArray detections_ = nullptr;
  VPIStream stream_ = nullptr;
  std::vector<std::function<void(const std::shared_ptr<NvidiaTagDetections>&)>>
      callbacks_;
};

}  // namespace apriltag
