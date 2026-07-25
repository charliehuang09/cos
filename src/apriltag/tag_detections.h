#pragma once

#include <array>
#include <cstddef>
#include <typeinfo>
#include <utility>
#include <vector>

#include <opencv2/core/types.hpp>

#include "control_loop/message.h"

namespace apriltag {

class TagDetections final : public control_loop::IMessage {
 public:
  struct tag_detection {
    int tag_id;
    std::array<cv::Point2d, 4> corners;
  };

  explicit TagDetections(std::vector<tag_detection> detections)
      : tag_detections(std::move(detections)) {}

  auto GetType() -> const std::type_info& override {
    return typeid(TagDetections);
  }
  auto GetSize() -> size_t override {
    return sizeof(*this) + tag_detections.capacity() * sizeof(tag_detection);
  }

  std::vector<tag_detection> tag_detections;
};

}  // namespace apriltag
