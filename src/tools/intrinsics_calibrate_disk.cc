#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "tools/charuco_calibration.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, detections_path, "",  // NOLINT
          "ChArUco detection JSON exported by calib_helper; calibrates "
          "without loading images");  // NOLINT
ABSL_FLAG(int, max_detections, 100,   // NOLINT
          "maximum calibration captures (0 means unlimited); "
          "randomly samples usable detections without replacement");  // NOLINT
ABSL_FLAG(int64_t, sampling_seed, -1,                                 // NOLINT
          "random sampling seed (0 through UINT32_MAX); -1 uses a random "
          "seed");                                    // NOLINT
ABSL_FLAG(std::string, intrinsics_output_path,        // NOLINT
          "intrinsics.json",                          // NOLINT
          "path for the generated intrinsics JSON");  // NOLINT

namespace {

using json = nlohmann::json;
using charuco_calibration::CalibrateCamera;
using charuco_calibration::DetectionFromJson;
using charuco_calibration::DetectionResult;
using charuco_calibration::HasEnoughCorners;
using charuco_calibration::IntrinsicsToJson;

auto WriteIntrinsicsToFile(const cv::Mat& camera_matrix,
                           const cv::Mat& dist_coeffs, const std::string& path)
    -> void {
  std::ofstream intrinsics_file(path);
  CHECK(intrinsics_file.is_open()) << "Failed to open " << path;
  const json intrinsics = IntrinsicsToJson(camera_matrix, dist_coeffs);
  intrinsics_file << intrinsics.dump(4) << '\n';

  std::cout << "Intrinsics:\n" << intrinsics.dump(4) << std::endl;
}

auto RunCalibration(const std::vector<DetectionResult>& detection_results,
                    cv::Size image_size) -> int {
  std::cout << "Calibrating with " << detection_results.size()
            << " captured frames" << std::endl;

  cv::Mat camera_matrix;
  cv::Mat dist_coeffs;
  std::optional<double> reprojection_error = CalibrateCamera(
      detection_results, image_size, &camera_matrix, &dist_coeffs);
  if (!reprojection_error.has_value()) {
    LOG(ERROR) << "No usable detections captured";
    return 1;
  }

  std::cout << "Reprojection error: " << *reprojection_error << std::endl;
  WriteIntrinsicsToFile(camera_matrix, dist_coeffs,
                        absl::GetFlag(FLAGS_intrinsics_output_path));
  return 0;
}

auto CalibrateDetectionsFile(const std::string& path, int max_detections)
    -> int {
  try {
    std::ifstream input(path);
    if (!input.is_open()) {
      throw std::runtime_error("Failed to open " + path);
    }
    json saved;
    input >> saved;
    const cv::Size image_size(saved.at("image_size").at("width").get<int>(),
                              saved.at("image_size").at("height").get<int>());
    if (image_size.width <= 0 || image_size.height <= 0) {
      throw std::runtime_error("Invalid image dimensions");
    }
    const auto& detections = saved.at("detections");
    if (!detections.is_array()) {
      throw std::runtime_error("detections must be an array");
    }
    std::vector<std::size_t> indices(detections.size());
    std::iota(indices.begin(), indices.end(), 0U);
    const int64_t configured_seed = absl::GetFlag(FLAGS_sampling_seed);
    const uint32_t seed = configured_seed < 0
                              ? std::random_device{}()
                              : static_cast<uint32_t>(configured_seed);
    std::cout << "Sampling seed: " << seed << std::endl;
    std::mt19937 generator(seed);
    std::shuffle(indices.begin(), indices.end(), generator);
    const std::size_t limit =
        max_detections == 0
            ? detections.size()
            : std::min(detections.size(),
                       static_cast<std::size_t>(max_detections));
    std::vector<DetectionResult> results;
    results.reserve(limit);
    // Visit a random permutation, deserializing only until enough usable
    // captures are loaded. Empty results do not consume the capture limit.
    for (const std::size_t index : indices) {
      if (stop::stop) {
        return 0;
      }
      DetectionResult result = DetectionFromJson(detections[index]);
      if (HasEnoughCorners(result)) {
        results.push_back(std::move(result));
        if (results.size() == limit) {
          break;
        }
      }
    }
    std::cout << "Selected " << results.size() << " of " << detections.size()
              << " saved detections from " << path << std::endl;
    return RunCalibration(results, image_size);
  } catch (const std::exception& error) {
    LOG(ERROR) << "Failed to calibrate saved detections: " << error.what();
    return 1;
  }
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  stop::RegisterHandler();

  const std::string detections_path = absl::GetFlag(FLAGS_detections_path);
  CHECK(!detections_path.empty()) << "--detections_path is required";
  const int max_detections = absl::GetFlag(FLAGS_max_detections);
  CHECK_GE(max_detections, 0) << "--max_detections must be nonnegative";
  CHECK_GE(absl::GetFlag(FLAGS_sampling_seed), -1);
  CHECK_LE(absl::GetFlag(FLAGS_sampling_seed),
           std::numeric_limits<uint32_t>::max());

  return CalibrateDetectionsFile(detections_path, max_detections);
}
