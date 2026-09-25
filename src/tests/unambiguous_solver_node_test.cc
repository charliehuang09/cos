#include "localization/unambiguous_solver_node.h"

#include <atomic>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <wpi/DataLogReader.h>
#include <wpi/MemoryBuffer.h>
#include <nlohmann/json.hpp>

#include "absl/base/log_severity.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "control_loop/thread_pool.h"
#include "simulation/simulation_position_sender_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(bool, reject_far_tags, true,                            // NOLINT
          "Reject tags and estimates that fail sanity checks.");  // NOLINT
ABSL_FLAG(int, poses_to_log, 20, "Stop after this many output poses.");
ABSL_FLAG(std::string, wpilog_path, "/root/unambiguous_solver_node_test.wpilog",
          "Where to save the replay's WPILOG.");
ABSL_FLAG(std::string, log_path, "/cos-logs/second_bot/chezychamps",
          "Directory containing front, left, and right camera replays.");

namespace {
void VerifyWPILog(const std::string& path) {
  auto buffer = wpi::MemoryBuffer::GetFile(path);
  CHECK(buffer.has_value()) << "Could not read WPILOG: " << path;
  wpi::log::DataLogReader reader(std::move(*buffer));
  CHECK(reader.IsValid()) << "Invalid WPILOG: " << path;

  std::unordered_map<int, std::string> entry_names;
  std::unordered_set<std::string> registered;
  std::unordered_map<std::string, int> written;
  std::unordered_map<std::string, int> nonempty_collections;
  for (const auto& record : reader) {
    if (record.IsStart()) {
      wpi::log::StartRecordData start;
      CHECK(record.GetStartData(&start));
      entry_names.emplace(start.entry, start.name);
      registered.emplace(start.name);
    } else if (!record.IsControl()) {
      const auto& name = entry_names.at(record.GetEntry());
      ++written[name];
      if (name == "pose/pose") {
        const auto bytes = record.GetRaw();
        CHECK_EQ(bytes.size(), wpi::Struct<frc::Pose3d>::GetSize());
        const auto pose = wpi::Struct<frc::Pose3d>::Unpack(bytes);
        CHECK(std::isfinite(pose.Translation().X().value()));
      } else if (name.ends_with("/tag_detections") ||
                 name.ends_with("/estimates")) {
        std::string_view serialized;
        CHECK(record.GetString(&serialized));
        const auto values = nlohmann::json::parse(std::string(serialized));
        CHECK(values.is_array());
        if (!values.empty()) {
          ++nonempty_collections[name];
          if (name.ends_with("/tag_detections")) {
            CHECK(values.front().contains("tag_id"));
            CHECK(values.front().contains("corners"));
            CHECK_EQ(values.front()["corners"].size(), 4U);
            CHECK(values.front()["corners"].front().contains("x"));
            CHECK(values.front()["corners"].front().contains("y"));
          } else {
            CHECK(values.front().contains("pos1"));
            const auto& estimate = values.front()["pos1"];
            CHECK(estimate.contains("tag_ids"));
            CHECK(estimate.contains("distances"));
            CHECK(estimate.contains("pose"));
            CHECK(estimate.contains("variance"));
            CHECK(estimate.contains("distance"));
            CHECK(estimate["pose"].contains("x"));
            CHECK(estimate["pose"].contains("y"));
            CHECK(estimate["pose"].contains("z"));
          }
        }
      } else if (name == "pose/tag_ids") {
        std::vector<std::int64_t> ids;
        CHECK(record.GetIntegerArray(&ids));
        CHECK(!ids.empty());
      } else if (name == "pose/distances") {
        std::vector<double> distances;
        CHECK(record.GetDoubleArray(&distances));
        CHECK(!distances.empty());
      } else if (name == "pose/variance") {
        double variance = 0;
        CHECK(record.GetDouble(&variance));
        CHECK(std::isfinite(variance));
      }
    }
  }

  std::vector<std::string> expected;
  for (std::string_view camera : {"front", "left", "right"}) {
    const std::string prefix = "second_bot_" + std::string(camera);
    for (std::string_view field : {"size", "timestamp"}) {
      expected.push_back(prefix + "/jpeg_buffer/" + std::string(field));
    }
    for (std::string_view field : {"fd", "pixel_format", "width", "height",
                                   "stride", "output_size", "timestamp"}) {
      expected.push_back(prefix + "/hardware_decoded_image/" +
                         std::string(field));
    }
    expected.push_back(prefix + "/hardware_decoded_image:latency/latency");
    expected.push_back(prefix + "/hardware_apriltag_detections:latency/latency");
    expected.push_back(prefix + "/hardware_apriltag_detections/tag_detections");
    expected.push_back(prefix +
                       "/hardware_apriltag_detections:multitag_solver/estimates");
  }
  for (std::string_view field : {"tag_ids", "pose", "distances", "variance"}) {
    expected.push_back("pose/" + std::string(field));
  }
  for (const auto& name : expected) {
    CHECK(registered.contains(name)) << "Missing WPILOG entry for " << name;
    if (!name.ends_with("/estimates")) {
      CHECK_GT(written[name], 0) << "Missing logged value for " << name;
    }
    LOG(INFO) << "WPILOG " << name << ": " << written[name] << " records";
  }
  CHECK(!nonempty_collections.empty());
  LOG(INFO) << "Verified replay WPILOG: " << path;
}
}  // namespace

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();
  control_loop::RioClock::EnableSimulation();

  control_loop::ControlLoop control_loop(1ms);
  const std::string wpilog_path = absl::GetFlag(FLAGS_wpilog_path);
  control_loop.EnableWPILog(wpilog_path);
  control_loop::ThreadPool thread_pool;
  std::atomic<int> logged_poses = 0;
  control_loop.SetMaxContext(1);
  control_loop.EnableLatencyLog();

  const std::filesystem::path replay_root = absl::GetFlag(FLAGS_log_path);
  const std::vector<std::string> replay_paths = {
      (replay_root / "second_bot_front").string(),
      (replay_root / "second_bot_left").string(),
      (replay_root / "second_bot_right").string()};
  const double replay_offset = camera::GetEarliestTimestamp(replay_paths);
  auto solver_node =
      std::make_shared<localization::UnambiguousSolverNode>("pose");
  solver_node->SetRejectFarTags(absl::GetFlag(FLAGS_reject_far_tags));
  control_loop.RegisterNode(solver_node);

  const std::array<std::string_view, 3> camera_names = {"front", "left", "right"};
  for (std::size_t i = 0; i < camera_names.size(); ++i) {
    const std::string name(camera_names[i]);
    const std::string prefix = "second_bot_" + name;
    const std::string jpeg_channel = prefix + "/jpeg_buffer";
    const std::string decoded_channel = prefix + "/hardware_decoded_image";
    const std::string detections_channel =
        prefix + "/hardware_apriltag_detections";
    const std::string config_path =
        "/root/constants/second_bot/" + name + "_camera.json";

    auto disk_camera_node = std::make_shared<camera::UVCDiskCameraNode>(
        replay_paths[i], jpeg_channel, replay_offset);
    control_loop.RegisterDependancyNode(disk_camera_node);

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>(
            jpeg_channel, "/stream", 4971 + static_cast<int>(i));
    control_loop.RegisterNode(jpeg_buffer_streamer_node);

    auto gpu_decode_node = std::make_shared<camera::NvjpegFdDecodeNode>(
        jpeg_channel, decoded_channel, thread_pool);
    control_loop.RegisterNode(gpu_decode_node);
    gpu_decode_node->EnableTiming(prefix + "/hardware_decoded_image:latency");

    auto gpu_apriltag_detector_node =
        std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
            decoded_channel, detections_channel, config_path, thread_pool);
    control_loop.RegisterNode(gpu_apriltag_detector_node);
    gpu_apriltag_detector_node->EnableTiming(
        prefix + "/hardware_apriltag_detections:latency");

    solver_node->AddCamera(detections_channel, camera::Intrinsics{config_path},
                           camera::Extrinsics{config_path}, control_loop);
  }
  solver_node->RegisterCallback(
      [&logged_poses](const control_loop::Context& context) -> void {
        auto pose = context->GetMessage<localization::PositionEstimateMessage>(
            "pose");
        if (pose != nullptr) {
          LOG(INFO) << *pose;
          if (++logged_poses >= absl::GetFlag(FLAGS_poses_to_log)) {
            stop::RequestStop();
          }
        }
      });

  auto simulation_position_sender_node =
      std::make_shared<simulation::SimulationPositionSenderNode>("pose");
  control_loop.RegisterNode(simulation_position_sender_node);

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();
  VerifyWPILog(wpilog_path);

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
