#include <opencv2/opencv.hpp>

#include "absl/log/check.h"
#include "absl/log/log.h"

#include "control_loop/timer.h"

#include "apriltag/gpu_apriltag_detector_lib.h"

auto main() -> int {
  std::string apriltag_path = "/root/apriltag.png";
  cv::Mat apriltag = cv::imread(apriltag_path, cv::IMREAD_GRAYSCALE);
  uint8_t* pixels = apriltag.data;
  uint height = apriltag.rows;
  uint width = apriltag.cols;
  CHECK(apriltag.step == static_cast<size_t>(apriltag.cols));
  auto detections = DetectAprilTag(
      apriltag::ImageView{
          .data = pixels, .stride = width, .height = height, .width = width},
      true);
  auto annotated_apriltag = apriltag.clone();
  DrawTagDetections(annotated_apriltag, detections);
  cv::imwrite("/root/annotated_apriltag.png", annotated_apriltag);
  constexpr int runs = 100;
  double average_run_time = 0.0;
  for (int i = 0; i < runs; i++) {
    control_loop::Timer timer;
    auto detections = DetectAprilTag(
        apriltag::ImageView{
            .data = pixels, .stride = width, .height = height, .width = width},
        false);
    average_run_time += timer.Stop().count();
  }
  LOG(INFO) << average_run_time / runs;
}
