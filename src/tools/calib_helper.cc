#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include <opencv2/imgcodecs.hpp>

#include "tools/charuco_calibration.h"
#include "utils/stop.h"

ABSL_FLAG(std::string, camera_folder, "",  // NOLINT
          "directory containing frames to detect "
          "(JPEG, PNG, BMP, TIFF)");  // NOLINT
ABSL_FLAG(std::string, detections_output_path, "detections.json",  // NOLINT
          "output JSON containing every frame's DetectionResult");  // NOLINT

namespace {

auto CameraFiles(const std::string& folder)
    -> std::vector<std::filesystem::path> {
  std::vector<std::filesystem::path> files;
  for (const auto& entry : std::filesystem::directory_iterator(folder)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::string extension = entry.path().extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char character) -> char {
                             return static_cast<char>(std::tolower(character));
                           });
    if (extension == ".jpg" || extension == ".jpeg" || extension == ".png" ||
        extension == ".bmp" || extension == ".tif" || extension == ".tiff") {
      files.push_back(entry.path());
    }
  }
  std::ranges::sort(files);
  return files;
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  stop::RegisterHandler();
  try {
    const std::string folder = absl::GetFlag(FLAGS_camera_folder);
    if (folder.empty()) {
      throw std::runtime_error("--camera_folder is required");
    }
    const auto files = CameraFiles(folder);
    if (files.empty()) {
      throw std::runtime_error("No image files in " + folder);
    }
    const std::filesystem::path output_path =
        absl::GetFlag(FLAGS_detections_output_path);
    for (const auto& file : files) {
      if (std::filesystem::weakly_canonical(file) ==
          std::filesystem::weakly_canonical(output_path)) {
        throw std::runtime_error("Output path would overwrite an input frame");
      }
    }
    std::ofstream output(output_path);
    if (!output.is_open()) {
      throw std::runtime_error("Failed to open " + output_path.string());
    }
    const auto detector =
        charuco_calibration::CreateDetector(charuco_calibration::CreateBoard());
    std::optional<cv::Size> image_size;
    std::size_t processed = 0;
    std::size_t usable = 0;
    // Stream results directly to disk, retaining only one decoded frame.
    output << "{\n  \"detections\": [\n";
    for (const auto& path : files) {
      if (stop::stop) {
        throw std::runtime_error("Detection export interrupted");
      }
      const cv::Mat frame = cv::imread(path.string(), cv::IMREAD_COLOR);
      if (frame.empty()) {
        throw std::runtime_error("Failed to decode " + path.string());
      }
      if (!image_size.has_value()) {
        image_size = frame.size();
      } else if (frame.size() != *image_size) {
        throw std::runtime_error("Inconsistent image dimensions in " +
                                 path.string());
      }
      const auto result =
          charuco_calibration::DetectCharucoBoard(frame, detector);
      auto saved = charuco_calibration::DetectionToJson(result);
      saved["filename"] = path.filename().string();
      if (processed > 0) {
        output << ",\n";
      }
      output << saved.dump();
      if (!output) {
        throw std::runtime_error("Failed to write " + output_path.string());
      }
      ++processed;
      usable += charuco_calibration::HasEnoughCorners(result) ? 1U : 0U;
      std::cout << "Detected " << processed << "/" << files.size() << ": "
                << path.filename() << " (" << result.charuco_corners.total()
                << " corners)" << std::endl;
    }
    output << "\n  ],\n  \"image_size\": "
           << nlohmann::json({{"width", image_size->width},
                              {"height", image_size->height}}).dump()
           << "\n}\n";
    output.close();
    if (!output) {
      throw std::runtime_error("Failed to write " + output_path.string());
    }
    std::cout << "Wrote " << processed << " detections (" << usable
              << " usable) to " << output_path << std::endl;
    return 0;
  } catch (const std::exception& error) {
    LOG(ERROR) << "Failed to export detections: " << error.what();
    return 1;
  }
}
