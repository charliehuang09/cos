#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

#include "localization/unambiguous_solver_node.h"

namespace calibration {

inline constexpr int64_t kMatchWindowNs = 10'000'000;

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

struct ReplayResult {
  size_t jpegs_processed = 0;
  std::vector<Observation> observations;
};

using ObservationGroup = std::vector<Observation>;

// Decimal seconds, up to nine fractional digits. Integer parsing preserves
// strict timestamp boundaries without floating point roundoff.
auto ParseCaptureTimestamp(std::string_view stem) -> int64_t;
auto EnumerateReplay(const std::vector<ReplayCamera>& cameras)
    -> std::vector<ReplayFrame>;

// The injected detector makes the lossless replay and branch selection usable
// independently of the GPU. Called once per frame in original timestamp order.
using FrameDetector = std::function<std::vector<localization::tag_detection_t>(
    const ReplayFrame&)>;
auto CollectObservations(const std::vector<ReplayCamera>& cameras,
                         const std::vector<ReplayFrame>& frames,
                         const FrameDetector& detector,
                         bool reject_far_tags = true,
                         const frc::AprilTagFieldLayout& layout =
                             localization::kApriltagLayout) -> ReplayResult;
auto ReplayObservations(const std::vector<ReplayCamera>& cameras,
                        bool reject_far_tags = true) -> ReplayResult;

// Greedily seed from the earliest available valid observation and take the
// earliest observation of each other camera within the strict 10 ms window.
// A camera occurs at most once per group. Non-reuse is the default.
auto MatchObservations(const std::vector<Observation>& observations,
                       bool reuse_frames = false)
    -> std::vector<ObservationGroup>;

struct GroupSplit {
  std::vector<ObservationGroup> training;
  std::vector<ObservationGroup> held_out;
};
// Every fifth chronological group is held out. With reuse enabled, overlapping
// groups stay together so a JPEG cannot leak between training and validation.
auto SplitGroups(const std::vector<ObservationGroup>& groups) -> GroupSplit;

}  // namespace calibration
