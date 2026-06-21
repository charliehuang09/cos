#pragma once
#include <functional>
#include <memory>
#include <nlohmann/json.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <vector>
#include "apriltag/apriltag_detector.h"

namespace apriltag {

class OpenCVApriltagDetectorNode : public IApriltagDetectorNode {
 public:
  explicit OpenCVApriltagDetectorNode(const nlohmann::json& intrinsics);

  void RegisterCallback(
      const std::function<void(std::shared_ptr<std::vector<tag_detection_t>>)>&
          callback) override;
  void Detect(const camera::DecodedJpegNvBuffer& frame,
              double timestamp) override;

 private:
  const cv::Mat camera_matrix_;
  const cv::Mat distortion_coefficients_;
  cv::aruco::ArucoDetector detector_;
  std::vector<
      std::function<void(std::shared_ptr<std::vector<tag_detection_t>>)>>
      callbacks_;
};

}  // namespace apriltag
