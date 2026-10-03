#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

#include "localization/unambiguous_solver_node.h"

namespace calibration {

struct ReplayCamera {
  std::string name;
  std::filesystem::path config_path;
  std::filesystem::path log_directory;
};

struct ReplayFrame {
  size_t camera;
  int64_t capture_ns;
  std::filesystem::path jpeg_path;
};

struct Observation {
  size_t frame_id;
  size_t camera;
  int64_t capture_ns;
  frc::Pose3d field_to_camera;  // T_F<-C in WPILib coordinates.
  frc::Pose3d field_to_robot;   // Original selected candidate, for auditing.
};

using ObservationGroup = std::vector<Observation>;

struct ReplayResult {
  size_t jpegs_processed = 0;
  std::vector<Observation> observations;
  std::vector<ObservationGroup> groups;
};

// Load cameras in batches from the latest camera start time. One frame per
// camera per batch; stop when any camera runs out of frames.
auto EnumerateReplay(const std::vector<ReplayCamera>& cameras)
    -> std::vector<std::vector<ReplayFrame>>;

// Detect every frame in a batch before jointly selecting the PnP candidates.
using FrameDetector = std::function<std::vector<localization::tag_detection_t>(
    const ReplayFrame&)>;
auto CollectObservations(
    const std::vector<ReplayCamera>& cameras,
    const std::vector<std::vector<ReplayFrame>>& batches,
    const FrameDetector& detector, bool reject_far_tags = true,
    const frc::AprilTagFieldLayout& layout = localization::kApriltagLayout)
    -> ReplayResult;
auto ReplayObservations(const std::vector<ReplayCamera>& cameras,
                        bool reject_far_tags = true) -> ReplayResult;

struct GroupSplit {
  std::vector<ObservationGroup> training;
  std::vector<ObservationGroup> held_out;
};
// Every fifth group is held out. Frames belong to exactly one group.
auto SplitGroups(const std::vector<ObservationGroup>& groups) -> GroupSplit;

}  // namespace calibration
