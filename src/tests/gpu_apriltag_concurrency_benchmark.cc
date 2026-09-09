#include <algorithm>
#include <array>
#include <barrier>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "apriltag/gpu_apriltag_detector.h"

ABSL_FLAG(std::string, input_folder, "", "JPEG recording directory");
ABSL_FLAG(int, instances, 4, "Concurrent detectors in this process");
ABSL_FLAG(int, max_images, 100, "Number of images to preload in filename order");
ABSL_FLAG(int, iterations, 10, "Timed passes through the preloaded recording");
ABSL_FLAG(int, decimate, 2, "Detector decimation factor");

namespace {
using Detections = std::vector<apriltag::ApriltagDetection>;
auto Canonical(const Detections& detections) {
  std::vector<std::array<int, 9>> result;
  for (const auto& detection : detections) {
    std::array<int, 9> entry{detection.id};
    for (int i = 0; i < 4; ++i) {
      entry[1 + 2 * i] = detection.quad.corners[i].row;
      entry[2 + 2 * i] = detection.quad.corners[i].col;
    }
    result.push_back(entry);
  }
  std::sort(result.begin(), result.end());
  return result;
}
auto View(const cv::Mat& image) -> apriltag::ImageView {
  return {.data = image.data, .stride = static_cast<int>(image.step),
          .height = image.rows, .width = image.cols};
}
}  // namespace

int main(int argc, char** argv) {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  const int instances = absl::GetFlag(FLAGS_instances);
  const int iterations = absl::GetFlag(FLAGS_iterations);
  const int max_images = absl::GetFlag(FLAGS_max_images);
  const int decimate = absl::GetFlag(FLAGS_decimate);
  CHECK_GT(instances, 0);
  CHECK_GT(iterations, 0);
  CHECK_GT(max_images, 0);
  CHECK_GT(decimate, 0);
  std::vector<std::filesystem::path> paths;
  for (const auto& file : std::filesystem::directory_iterator(
           absl::GetFlag(FLAGS_input_folder))) {
    const auto extension = file.path().extension();
    if (extension == ".jpg" || extension == ".jpeg") paths.push_back(file.path());
  }
  std::sort(paths.begin(), paths.end());
  if (paths.size() > static_cast<size_t>(max_images)) paths.resize(max_images);
  CHECK(!paths.empty());
  std::vector<cv::Mat> images;
  for (const auto& path : paths) {
    images.push_back(cv::imread(path.string(), cv::IMREAD_GRAYSCALE));
    CHECK(!images.back().empty()) << path;
    CHECK(images.back().size() == images.front().size());
  }
  const auto size = images.front().size();
  CHECK_EQ(size.width % (4LL * decimate), 0);
  CHECK_EQ(size.height % (4LL * decimate), 0);
  std::vector<int> ids(32);
  std::iota(ids.begin(), ids.end(), 1);
  std::vector<std::unique_ptr<apriltag::GPUApriltagDetector>> detectors;
  for (int i = 0; i < instances; ++i) {
    detectors.push_back(std::make_unique<apriltag::GPUApriltagDetector>(
        size.width, size.height, ids, decimate));
  }
  std::vector<Detections> expected;
  for (const auto& image : images) expected.push_back(detectors.front()->Detect(View(image)));

  using Clock = std::chrono::steady_clock;
  Clock::time_point begin, end;
  // Completion runs before any participant is released, so no detector work
  // can start before the shared timer or finish after it stops.
  std::barrier start(instances + 1, [&]() noexcept { begin = Clock::now(); });
  std::barrier finish(instances + 1, [&]() noexcept { end = Clock::now(); });
  std::vector<double> seconds(instances);
  std::vector<std::thread> workers;
  for (int worker = 0; worker < instances; ++worker) {
    workers.emplace_back([&, worker] {
      auto& detector = *detectors[worker];
      // Warm and grow each workspace before timing. Independent detectors
      // share only read-only images and the production geometry worker pool.
      for (const auto& image : images) detector.Detect(View(image));
      std::vector<Detections> results(images.size());
      start.arrive_and_wait();
      for (int iteration = 0; iteration < iterations; ++iteration) {
        for (size_t frame = 0; frame < images.size(); ++frame) {
          results[frame] = detector.Detect(View(images[frame]));
        }
      }
      seconds[worker] = std::chrono::duration<double>(Clock::now() - begin).count();
      finish.arrive_and_wait();
      for (size_t frame = 0; frame < images.size(); ++frame) {
        CHECK(Canonical(results[frame]) == Canonical(expected[frame]))
            << "Detector " << worker << ", frame " << frame;
      }
    });
  }
  start.arrive_and_wait();
  finish.arrive_and_wait();
  const double elapsed = std::chrono::duration<double>(end - begin).count();
  for (auto& worker : workers) worker.join();
  const double frames = double(images.size()) * iterations;
  std::cout << "input=" << size.width << 'x' << size.height
            << " decimate=" << decimate << " instances=" << instances
            << " frames_per_detector=" << frames << " elapsed_s=" << elapsed
            << " aggregate_fps=" << instances * frames / elapsed
            << " per_camera_fps=" << frames / elapsed << '\n';
  for (int i = 0; i < instances; ++i) {
    std::cout << "detector=" << i << " fps=" << frames / seconds[i]
              << " mean_ms=" << 1000 * seconds[i] / frames << '\n';
  }
  std::cout << "All detector outputs match the sequential reference.\n";
}
