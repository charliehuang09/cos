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
ABSL_FLAG(int, decimate, 1,
          "Integer image reduction factor; dimensions must be divisible by 4 * factor");

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  std::string apriltag_path = absl::GetFlag(FLAGS_image_path);
  cv::Mat apriltag = cv::imread(apriltag_path, cv::IMREAD_GRAYSCALE);
  CHECK(!apriltag.empty()) << "Failed to read " << apriltag_path;
  const int factor = absl::GetFlag(FLAGS_decimate);
  CHECK_GE(factor, 1);
  CHECK_EQ(apriltag.cols % (4LL * factor), 0);
  CHECK_EQ(apriltag.rows % (4LL * factor), 0);
  const int width = apriltag.cols / factor;
  const int height = apriltag.rows / factor;

  auto detector = apriltag::GPUApriltagDetector(width, height);
  cv::Mat reduced;
  auto detect = [&](bool debug = false) {
    if (factor > 1) {
      cv::resize(apriltag, reduced, cv::Size(width, height),
                 0, 0, cv::INTER_AREA);
    } else {
      reduced = apriltag;
    }
    apriltag::ImageView view{
        .data = reduced.data, .stride = static_cast<int>(reduced.step),
        .height = height, .width = width};
    auto detections = detector.Detect(view, debug);
    // Draw detections in the original image's pixel-center coordinates.
    if (factor > 1) {
      for (auto& detection : detections) {
        for (auto& point : detection.quad.corners) {
          point.row = (point.row + 0.5f) * factor - 0.5f;
          point.col = (point.col + 0.5f) * factor - 0.5f;
        }
      }
    }
    return detections;
  };
  auto detections = detect(true);
  LOG(INFO) << "Decimation factor: " << factor << "; detector image: "
            << width << "x" << height << "; detected tags: " << detections.size();
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
    detections = detect();
    average_run_time += timer.Stop().count();
  }
  LOG(INFO) << "Average time (ms, including resize): "
            << 1000 * average_run_time / runs;
}
