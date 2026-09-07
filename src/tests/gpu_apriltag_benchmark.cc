#include <numeric>
#include <memory>
#include <iterator>
#include <opencv2/imgproc.hpp>
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

ABSL_FLAG(int, decimate, 1, "Integer image reduction factor; >1 compares against full resolution");
ABSL_FLAG(int, iterations, 3, "Timed repetitions per image (after warmup)");

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  std::string input_folder = absl::GetFlag(FLAGS_input_folder);

  std::vector<int> target_ids(32);
  std::iota(target_ids.begin(), target_ids.end(), 1);
  const int factor = absl::GetFlag(FLAGS_decimate);
  const int iterations = absl::GetFlag(FLAGS_iterations);
  CHECK_GE(factor, 1);
  CHECK_GT(iterations, 0);
  std::unique_ptr<apriltag::GPUApriltagDetector> detector;
  std::unique_ptr<apriltag::GPUApriltagDetector> baseline;
  cv::Size input_size;

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
  double baseline_time = 0;
  size_t baseline_tags = 0, matched_ids = 0, same_count_images = 0;
  size_t same_ids_images = 0;
  for (size_t index = 0; index < images.size();
       index += absl::GetFlag(FLAGS_sample_step)) {
    if (stop::StopRequested()) {
      break;
    }

    cv::Mat apriltag = cv::imread(images[index].string(), cv::IMREAD_GRAYSCALE);
    CHECK(!apriltag.empty()) << images[index];
    if (!detector) {
      input_size = apriltag.size();
      CHECK_EQ(input_size.width % (4LL * factor), 0);
      CHECK_EQ(input_size.height % (4LL * factor), 0);
      detector = std::make_unique<apriltag::GPUApriltagDetector>(
          input_size.width / factor, input_size.height / factor, target_ids);
      if (factor > 1) baseline = std::make_unique<apriltag::GPUApriltagDetector>(
          input_size.width, input_size.height, target_ids);
    }
    CHECK(apriltag.size() == input_size);
    auto view = [](const cv::Mat& image) {
      return apriltag::ImageView{.data = image.data,
          .stride = static_cast<int>(image.step),
          .height = image.rows, .width = image.cols};
    };
    cv::Mat reduced;
    std::vector<apriltag::ApriltagDetection> detections, original;
    auto run_reduced = [&] {
      if (factor > 1) {
        cv::resize(apriltag, reduced,
            cv::Size(input_size.width / factor, input_size.height / factor),
            0, 0, cv::INTER_AREA);
      } else {
        reduced = apriltag;
      }
      detections = detector->Detect(view(reduced));
    };
    auto run_original = [&] { original = baseline->Detect(view(apriltag)); };
    // Warm each image before timing; exclude image loading and detector setup.
    run_reduced();
    if (baseline) run_original();
    double reduced_seconds = 0, original_seconds = 0;
    for (int repetition = 0; repetition < iterations; ++repetition) {
      auto time_reduced = [&] {
        control_loop::Timer timer;
        run_reduced();
        reduced_seconds += timer.Stop().count();
      };
      auto time_original = [&] {
        if (!baseline) return;
        control_loop::Timer timer;
        run_original();
        original_seconds += timer.Stop().count();
      };
      // Alternate execution order to reduce order-dependent timing bias.
      if ((valid_images + repetition) % 2 == 0) {
        time_original();
        time_reduced();
      } else {
        time_reduced();
        time_original();
      }
    }
    valid_images++;
    total_time += reduced_seconds / iterations;
    baseline_time += original_seconds / iterations;
    detected_tags_sum += detections.size();
    if (baseline) {
      baseline_tags += original.size();
      same_count_images += original.size() == detections.size();
      std::vector<int> original_ids, reduced_ids, common;
      for (const auto& detection : original) original_ids.push_back(detection.id);
      for (const auto& detection : detections) reduced_ids.push_back(detection.id);
      std::sort(original_ids.begin(), original_ids.end());
      std::sort(reduced_ids.begin(), reduced_ids.end());
      std::set_intersection(original_ids.begin(), original_ids.end(),
          reduced_ids.begin(), reduced_ids.end(), std::back_inserter(common));
      matched_ids += common.size();
      same_ids_images += original_ids == reduced_ids;
    }
    // Return exported coordinates to the original image's pixel-center space.
    for (auto& detection : detections) {
      for (auto& point : detection.quad.corners) {
        point.row = (point.row + 0.5f) * factor - 0.5f;
        point.col = (point.col + 0.5f) * factor - 0.5f;
      }
    }
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
  LOG(INFO) << "Decimation factor: " << factor;
  LOG(INFO) << "Average time (ms, including resize): " << 1000 * total_time / valid_images;
  if (baseline) {
    LOG(INFO) << "Full-resolution tags: " << baseline_tags;
    LOG(INFO) << "Full-resolution average time (ms): " << 1000 * baseline_time / valid_images;
    LOG(INFO) << "Speedup: " << baseline_time / total_time << "x";
    LOG(INFO) << "Time reduction (%): " << 100 * (1 - total_time / baseline_time);
    LOG(INFO) << "Images with equal counts: " << same_count_images << "/" << valid_images;
    LOG(INFO) << "Images with identical ID multisets: " << same_ids_images << "/" << valid_images;
    LOG(INFO) << "Matched baseline IDs: " << matched_ids << "/" << baseline_tags;
    if (baseline_tags > 0) {
      LOG(INFO) << "Tag count ratio (%): " << 100.0 * detected_tags_sum / baseline_tags;
    }
  }
}
