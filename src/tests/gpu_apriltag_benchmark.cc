#include <opencv2/imgcodecs.hpp>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/gpu_apriltag_detector.h"
#include "control_loop/timer.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, input_folder, "",           // NOLINT
          "Input folder filled with .jpg files");  // NOLINT

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  std::string input_folder = absl::GetFlag(FLAGS_input_folder);

  auto detector = apriltag::GPUApriltagDetector(1280, 800);

  CHECK(input_folder != "");

  int valid_images = 0;
  int detected_tags_sum = 0;
  double total_time = 0;
  for (const auto& entry : std::filesystem::directory_iterator(input_folder)) {
    if (stop::StopRequested()) {
      break;
    }

    std::string extension = entry.path().extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char c) -> char {
                             return static_cast<char>(std::tolower(c));
                           });
    if (extension != ".jpg" && extension != ".jpeg") {
      continue;
    }

    cv::Mat apriltag = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
    uint8_t* pixels = apriltag.data;
    int height = apriltag.rows;
    int width = apriltag.cols;
    apriltag::ImageView apriltag_view{
        .data = pixels, .stride = width, .height = height, .width = width};
    control_loop::Timer timer;
    auto detections = detector.Detect(apriltag_view);
    valid_images++;
    total_time += timer.Stop().count();
    detected_tags_sum += detections.size();
  }
  LOG(INFO) << "Detected tags sum: " << detected_tags_sum;
  LOG(INFO) << "Average time: " << total_time / valid_images;
}
