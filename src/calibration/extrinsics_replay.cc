#include "calibration/extrinsics_replay.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "control_loop/thread_pool.h"

namespace calibration {

auto ParseCaptureTimestamp(std::string_view stem) -> int64_t {
  if (stem.empty()) throw std::invalid_argument("Empty capture timestamp");
  int64_t seconds = 0, fraction = 0;
  bool decimal = false;
  size_t digits = 0, whole_digits = 0;
  constexpr auto max = std::numeric_limits<int64_t>::max();
  for (char c : stem) {
    if (c == '.' && !decimal) {
      decimal = true;
      continue;
    }
    if (c < '0' || c > '9')
      throw std::invalid_argument("Invalid capture timestamp: " + std::string(stem));
    const int digit = c - '0';
    if (!decimal) {
      if (seconds > (max / 1'000'000'000 - digit) / 10)
        throw std::out_of_range("Capture timestamp overflow");
      seconds = seconds * 10 + digit;
      ++whole_digits;
    } else {
      if (++digits > 9) throw std::invalid_argument("Timestamp exceeds ns precision");
      fraction = fraction * 10 + digit;
    }
  }
  if (whole_digits == 0 || (decimal && digits == 0))
    throw std::invalid_argument("Invalid capture timestamp: " + std::string(stem));
  for (; digits < 9; ++digits) fraction *= 10;
  if (seconds > (max - fraction) / 1'000'000'000)
    throw std::out_of_range("Capture timestamp overflow");
  return seconds * 1'000'000'000 + fraction;
}

auto EnumerateReplay(const std::vector<ReplayCamera>& cameras)
    -> std::vector<ReplayFrame> {
  std::vector<ReplayFrame> frames;
  std::unordered_set<std::string> names;
  for (size_t i = 0; i < cameras.size(); ++i) {
    if (!names.insert(cameras[i].name).second)
      throw std::invalid_argument("Duplicate camera: " + cameras[i].name);
    for (const auto& entry : std::filesystem::directory_iterator(cameras[i].log_directory)) {
      if (!entry.is_regular_file()) continue;
      auto extension = entry.path().extension().string();
      std::ranges::transform(extension, extension.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
      });
      if (extension != ".jpg" && extension != ".jpeg") continue;
      // A JPEG with an invalid timestamp is an input error, never silently lost.
      frames.push_back({i, ParseCaptureTimestamp(entry.path().stem().string()), entry.path()});
    }
  }
  std::ranges::sort(frames, [](const auto& a, const auto& b) {
    if (a.capture_ns != b.capture_ns) return a.capture_ns < b.capture_ns;
    if (a.camera != b.camera) return a.camera < b.camera;
    return a.jpeg_path < b.jpeg_path;
  });
  if (frames.empty()) throw std::invalid_argument("Replay contains no JPEGs");
  return frames;
}

auto CollectObservations(const std::vector<ReplayCamera>& cameras,
                         const std::vector<ReplayFrame>& frames,
                         const FrameDetector& detector, bool reject_far_tags,
                         const frc::AprilTagFieldLayout& layout) -> ReplayResult {
  std::vector<std::unique_ptr<localization::MultiTagSolverNode>> pnp;
  for (const auto& camera : cameras) {
    pnp.push_back(std::make_unique<localization::MultiTagSolverNode>(
        "", "", camera::Intrinsics{camera.config_path},
        camera::Extrinsics{camera.config_path}, layout));
  }
  localization::UnambiguousSolverNode selector{"", layout};
  ReplayResult result;
  for (size_t id = 0; id < frames.size(); ++id) {
    const auto& frame = frames[id];
    if (frame.camera >= cameras.size() || frame.capture_ns < 0 ||
        (id && frame.capture_ns < frames[id - 1].capture_ns))
      throw std::invalid_argument("Replay frames must be in capture order with valid camera indices");
    const auto detections = detector(frame);
    ++result.jpegs_processed;
    auto candidates = pnp[frame.camera]->AmbiguousSolve(detections, reject_far_tags);
    if (!candidates) continue;
    // Run selection throughout the entire replay, including isolated frames.
    // Previous-pose state advances on each valid frame, before any matching.
    auto solution = selector.SolveSelected({&*candidates}, reject_far_tags);
    if (!solution) continue;
    const auto& selected = solution->selected.front().estimate;
    if (!selected.field_to_camera)
      throw std::logic_error("Selected PnP candidate has no camera pose");
    if (!selected.field_to_camera->ToMatrix().allFinite() ||
        !selected.pose.ToMatrix().allFinite()) continue;
    result.observations.push_back({id, frame.camera, frame.capture_ns,
                                  *selected.field_to_camera, selected.pose});
  }
  return result;
}

auto ReplayObservations(const std::vector<ReplayCamera>& cameras,
                        bool reject_far_tags) -> ReplayResult {
  const auto frames = EnumerateReplay(cameras);
  control_loop::ThreadPool pool{1};
  camera::NvjpegFdDecodeNode decoder{"", "", pool};
  std::vector<std::unique_ptr<apriltag::NvidiaApriltagDetectorNode>> detectors;
  for (const auto& camera : cameras) {
    detectors.push_back(std::make_unique<apriltag::NvidiaApriltagDetectorNode>(
        "", "", camera.config_path.string(), pool));
  }
  return CollectObservations(cameras, frames, [&](const ReplayFrame& frame) {
    std::ifstream input(frame.jpeg_path, std::ios::binary | std::ios::ate);
    const std::streamsize size = input.tellg();
    if (!input || size <= 0)
      throw std::runtime_error("Cannot read JPEG: " + frame.jpeg_path.string());
    camera::JpegBuffer jpeg{static_cast<size_t>(size), frame.capture_ns / 1e9};
    if (jpeg.ptr == nullptr) throw std::bad_alloc{};
    input.seekg(0);
    input.read(reinterpret_cast<char*>(jpeg.ptr), size);
    if (!input) throw std::runtime_error("Cannot read JPEG: " + frame.jpeg_path.string());
    auto decoded = decoder.Decode(jpeg);
    if (!decoded) throw std::runtime_error("Cannot decode JPEG: " + frame.jpeg_path.string());
    return detectors[frame.camera]->Detect(*decoded);
  }, reject_far_tags);
}

auto MatchObservations(const std::vector<Observation>& observations, bool reuse_frames)
    -> std::vector<ObservationGroup> {
  auto sorted = observations;
  std::ranges::sort(sorted, [](const auto& a, const auto& b) {
    if (a.capture_ns != b.capture_ns) return a.capture_ns < b.capture_ns;
    return a.frame_id < b.frame_id;
  });
  std::unordered_set<size_t> frame_ids;
  for (const auto& observation : sorted) {
    if (observation.capture_ns < 0 || !frame_ids.insert(observation.frame_id).second)
      throw std::invalid_argument("Observations need nonnegative timestamps and unique frame IDs");
  }
  std::vector<bool> used(sorted.size(), false);
  std::vector<ObservationGroup> groups;
  for (size_t i = 0; i < sorted.size(); ++i) {
    if (!reuse_frames && used[i]) continue;
    std::vector<size_t> indices{i};
    std::unordered_set<size_t> cameras{sorted[i].camera};
    for (size_t j = i + 1; j < sorted.size(); ++j) {
      if (sorted[j].capture_ns - sorted[i].capture_ns >= kMatchWindowNs) break;
      if ((!reuse_frames && used[j]) || !cameras.insert(sorted[j].camera).second) continue;
      indices.push_back(j);
    }
    if (indices.size() < 2) continue;
    ObservationGroup group;
    for (size_t index : indices) {
      group.push_back(sorted[index]);
      used[index] = true;
    }
    groups.push_back(std::move(group));
  }
  return groups;
}

auto SplitGroups(const std::vector<ObservationGroup>& groups) -> GroupSplit {
  std::vector<size_t> parent(groups.size());
  std::iota(parent.begin(), parent.end(), 0);
  auto root = [&](size_t i) {
    while (parent[i] != i) {
      parent[i] = parent[parent[i]];
      i = parent[i];
    }
    return i;
  };
  std::unordered_map<size_t, size_t> frame_group;
  for (size_t i = 0; i < groups.size(); ++i) {
    for (const auto& observation : groups[i]) {
      auto [it, inserted] = frame_group.emplace(observation.frame_id, i);
      if (!inserted) parent[root(i)] = root(it->second);
    }
  }
  std::unordered_map<size_t, size_t> component_ids;
  GroupSplit split;
  for (size_t i = 0; i < groups.size(); ++i) {
    auto [it, inserted] = component_ids.emplace(root(i), component_ids.size());
    (void)inserted;
    auto& destination = it->second % 5 == 4 ? split.held_out : split.training;
    destination.push_back(groups[i]);
  }
  return split;
}

}  // namespace calibration
