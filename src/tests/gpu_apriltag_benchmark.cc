#include <numeric>
#include <fstream>
#include <algorithm>
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
ABSL_FLAG(int, max_images, 0, "Maximum images to process; 0 means all");
ABSL_FLAG(int, sample_step, 1, "Process every Nth image in filename order");
ABSL_FLAG(std::string, detections_output, "", "Optional per-image detection TSV");

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  std::string input_folder = absl::GetFlag(FLAGS_input_folder);

  std::vector<int> target_ids(32);
  std::iota(target_ids.begin(), target_ids.end(), 1);
  auto detector = apriltag::GPUApriltagDetector(1280, 800, target_ids);

  CHECK(input_folder != "");
  CHECK_GE(absl::GetFlag(FLAGS_max_images), 0);
  CHECK_GT(absl::GetFlag(FLAGS_sample_step), 0);
  std::ofstream output;
  if (!absl::GetFlag(FLAGS_detections_output).empty()) {
    output.open(absl::GetFlag(FLAGS_detections_output));
    CHECK(output.is_open());
  }

  std::vector<std::filesystem::path> images;
  for (const auto& entry : std::filesystem::directory_iterator(input_folder)) {
    std::string extension = entry.path().extension().string();
    std::ranges::transform(extension, extension.begin(),
        [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (extension == ".jpg" || extension == ".jpeg") images.push_back(entry.path());
  }
  std::sort(images.begin(), images.end());

  int valid_images = 0;
  int detected_tags_sum = 0;
  double total_time = 0;
  for (size_t index = 0; index < images.size();
       index += absl::GetFlag(FLAGS_sample_step)) {
    if (stop::StopRequested()) {
      break;
    }

    cv::Mat apriltag = cv::imread(images[index].string(), cv::IMREAD_GRAYSCALE);
    CHECK(!apriltag.empty()) << images[index];
    uint8_t* pixels = apriltag.data;
    int height = apriltag.rows;
    int width = apriltag.cols;
    apriltag::ImageView apriltag_view{
        .data = pixels, .stride = static_cast<int>(apriltag.step),
        .height = height, .width = width};
    control_loop::Timer timer;
    auto detections = detector.Detect(apriltag_view);
    valid_images++;
    total_time += timer.Stop().count();
    detected_tags_sum += detections.size();
    if (output.is_open()) {
      std::sort(detections.begin(), detections.end(), [](const auto& a, const auto& b) {
        if (a.id != b.id) return a.id < b.id;
        return std::lexicographical_compare(
            a.quad.corners.begin(), a.quad.corners.end(),
            b.quad.corners.begin(), b.quad.corners.end(), [](auto x, auto y) {
              return x.row != y.row ? x.row < y.row : x.col < y.col;
            });
      });
      output << images[index].filename().string();
      for (const auto& detection : detections) {
        output << '\t' << detection.id;
        for (const auto& point : detection.quad.corners) {
          output << ',' << point.row << ',' << point.col;
        }
      }
      output << '\n';
    }
    if (absl::GetFlag(FLAGS_max_images) > 0 &&
        valid_images >= absl::GetFlag(FLAGS_max_images)) break;
  }
  CHECK_GT(valid_images, 0);
  if (output.is_open()) {
    output.flush();
    CHECK(output.good());
  }
  LOG(INFO) << "Processed images: " << valid_images;
  LOG(INFO) << "Detected tags sum: " << detected_tags_sum;
  LOG(INFO) << "Average time: " << total_time / valid_images;
}
