#include "calibration/extrinsics_replay.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>

#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "control_loop/thread_pool.h"

namespace calibration {

auto EnumerateReplay(const std::vector<ReplayCamera>& cameras)
    -> std::vector<std::vector<ReplayFrame>> {
  std::vector<std::vector<std::pair<std::filesystem::path, double>>> files;
  double start = -std::numeric_limits<double>::infinity();
  for (const auto& camera : cameras) {
    files.push_back(camera::GetTimestampedJpegs(camera.log_directory));
    if (files.back().empty())
      throw std::invalid_argument("No timestamped JPEGs for camera: " +
                                  camera.name);
    start = std::max(start, files.back().front().second);
  }
  if (files.empty())
    throw std::invalid_argument("No replay cameras");

  // Skip each camera's early frames to start within their common time range.
  size_t count = std::numeric_limits<size_t>::max();
  for (auto& camera_files : files) {
    const auto first = std::ranges::lower_bound(
        camera_files, start, {}, [](const auto& file) { return file.second; });
    camera_files.erase(camera_files.begin(), first);
    count = std::min(count, camera_files.size());
  }
  if (count == 0)
    throw std::invalid_argument("Camera replays do not overlap");

  // Advance every camera once per batch, stopping when the first camera ends.
  std::vector<std::vector<ReplayFrame>> batches(count);
  for (size_t i = 0; i < count; ++i) {
    for (size_t camera = 0; camera < files.size(); ++camera) {
      const auto& [path, timestamp] = files[camera][i];
      batches[i].push_back({camera, std::llround(timestamp * 1e9), path});
    }
  }
  return batches;
}

auto CollectObservations(const std::vector<ReplayCamera>& cameras,
                         const std::vector<std::vector<ReplayFrame>>& batches,
                         const FrameDetector& detector, bool reject_far_tags,
                         const frc::AprilTagFieldLayout& layout)
    -> ReplayResult {
  std::vector<std::unique_ptr<localization::MultiTagSolverNode>> pnp;
  for (const auto& camera : cameras) {
    pnp.push_back(std::make_unique<localization::MultiTagSolverNode>(
        "", "", camera::Intrinsics{camera.config_path},
        camera::Extrinsics{camera.config_path}, layout));
  }
  localization::UnambiguousSolverNode selector{"", layout};
  ReplayResult result;
  size_t frame_id = 0;
  for (const auto& batch : batches) {
    std::vector<std::optional<localization::ambiguous_estimate_t>> candidates;
    for (const auto& frame : batch) {
      candidates.push_back(
          pnp[frame.camera]->AmbiguousSolve(detector(frame), reject_far_tags));
      ++result.jpegs_processed;
    }
    std::vector<localization::ambiguous_estimate_t*> estimates;
    for (auto& candidate : candidates)
      estimates.push_back(candidate ? &*candidate : nullptr);
    const auto solution = selector.SolveSelected(estimates, reject_far_tags);
    ObservationGroup group;
    if (solution) {
      for (const auto& selected : solution->selected) {
        const auto& frame = batch[selected.input_index];
        const auto& estimate = selected.estimate;
        if (!estimate.field_to_camera)
          continue;
        Observation observation{.frame_id = frame_id + selected.input_index,
                                .camera = frame.camera,
                                .capture_ns = frame.capture_ns,
                                .field_to_camera = *estimate.field_to_camera,
                                .field_to_robot = estimate.pose};
        result.observations.push_back(observation);
        group.push_back(observation);
      }
    }
    if (group.size() >= 2)
      result.groups.push_back(std::move(group));
    frame_id += batch.size();
  }
  return result;
}

auto ReplayObservations(const std::vector<ReplayCamera>& cameras,
                        bool reject_far_tags) -> ReplayResult {
  const auto batches = EnumerateReplay(cameras);
  control_loop::ThreadPool pool{1};
  camera::NvjpegFdDecodeNode decoder{"", "", pool};
  std::vector<std::unique_ptr<apriltag::NvidiaApriltagDetectorNode>> detectors;
  for (const auto& camera : cameras) {
    detectors.push_back(std::make_unique<apriltag::NvidiaApriltagDetectorNode>(
        "", "", camera.config_path.string(), pool));
  }
  return CollectObservations(
      cameras, batches,
      [&](const ReplayFrame& frame) {
        std::ifstream input(frame.jpeg_path, std::ios::binary | std::ios::ate);
        const std::streamsize size = input.tellg();
        if (!input || size <= 0)
          throw std::runtime_error("Cannot read JPEG: " +
                                   frame.jpeg_path.string());
        camera::JpegBuffer jpeg{static_cast<size_t>(size),
                                frame.capture_ns / 1e9};
        if (jpeg.ptr == nullptr)
          throw std::bad_alloc{};
        input.seekg(0);
        input.read(reinterpret_cast<char*>(jpeg.ptr), size);
        if (!input)
          throw std::runtime_error("Cannot read JPEG: " +
                                   frame.jpeg_path.string());
        auto decoded = decoder.Decode(jpeg);
        if (!decoded)
          throw std::runtime_error("Cannot decode JPEG: " +
                                   frame.jpeg_path.string());
        return detectors[frame.camera]->Detect(*decoded);
      },
      reject_far_tags);
}

auto SplitGroups(const std::vector<ObservationGroup>& groups) -> GroupSplit {
  GroupSplit split;
  for (size_t i = 0; i < groups.size(); ++i) {
    auto& destination = i % 5 == 4 ? split.held_out : split.training;
    destination.push_back(groups[i]);
  }
  return split;
}

}  // namespace calibration
