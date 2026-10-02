#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

namespace gamepiece {

struct LabeledBoundingBox {
  cv::Rect bounds;
  std::string label;
  int class_id = -1;
  float confidence = 0.0F;
};

class ObjectDetector {
 public:
  virtual ~ObjectDetector() = default;

  virtual auto Detect(const cv::cuda::GpuMat& image)
      -> std::vector<LabeledBoundingBox> = 0;
};

}  // namespace gamepiece
