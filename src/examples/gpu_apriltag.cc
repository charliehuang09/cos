#include <opencv2/opencv.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "control_loop/timer.h"

#include "apriltag/gpu_apriltag_detector_lib.h"

ABSL_FLAG(std::string, image_path, "/root/apriltag.png",               // NOLINT
          "Apriltag image, width and height must be divisible by 4");  // NOLINT

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  std::string apriltag_path = absl::GetFlag(FLAGS_image_path);
  cv::Mat apriltag = cv::imread(apriltag_path, cv::IMREAD_GRAYSCALE);
  uint8_t* pixels = apriltag.data;
  int height = apriltag.rows;
  int width = apriltag.cols;
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
