#include <opencv2/opencv.hpp>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "apriltag/gpu_apriltag_detector.h"

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
  apriltag::ImageView apriltag_view{
      .data = pixels, .stride = width, .height = height, .width = width};

  auto detector = apriltag::GPUApriltagDetector(width, height);
  auto detections = detector.Detect(apriltag_view, true);
  auto annotated_apriltag = apriltag.clone();
  DrawTagDetections(annotated_apriltag, detections);
  const std::filesystem::path log_path = "/root/apriltag_logs";
  std::error_code create_directory_error;
  std::filesystem::create_directories(log_path, create_directory_error);
  CHECK(!create_directory_error)
      << "Failed to create " << log_path << ": "
      << create_directory_error.message();
  CHECK(cv::imwrite((log_path / "annotated_apriltag.png").string(),
                    annotated_apriltag));
  detector.WriteLogImages(log_path);

  constexpr int runs = 250;
  double average_run_time = 0.0;
  for (int i = 0; i < runs; i++) {
    control_loop::Timer timer;
    detections = detector.Detect(apriltag_view);
    average_run_time += timer.Stop().count();
  }
  LOG(INFO) << average_run_time / runs;
}
