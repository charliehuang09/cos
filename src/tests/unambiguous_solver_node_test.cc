#include "localization/unambiguous_solver_node.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numbers>
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
#include "camera/nvjpeg_decode_node.h"
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
ABSL_FLAG(std::string, wpilog_path,
          "/root/unambiguous_solver_node_test.wpilog",  // NOLINT
          "Where to save the replay's WPILOG.");        // NOLINT
ABSL_FLAG(std::string, camera_config_directory, "/root/constants/second_bot",
          "Directory containing the camera JSON configurations.");
ABSL_FLAG(std::string, disagreement_csv_path, "",
          "Optional CSV of camera pairs selected by the live solver.");
ABSL_FLAG(
    std::string, log_path, "/cos-logs/second_bot/chezychamps",       // NOLINT
    "Directory containing front, left, and right camera replays.");  // NOLINT

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
  control_loop.SetMaxContext(1);
  control_loop.EnableLatencyLog();

  const std::filesystem::path replay_root = absl::GetFlag(FLAGS_log_path);
  const std::vector<std::string> replay_paths = {
      (replay_root / "front").string(), (replay_root / "left").string(),
      (replay_root / "right").string()};
  const double replay_offset = camera::GetEarliestTimestamp(replay_paths);
  auto solver_node =
      std::make_shared<localization::UnambiguousSolverNode>("pose");
  const std::array<std::string_view, 3> camera_names = {"front", "left",
                                                      "right"};
  std::ofstream disagreement_csv;
  const std::string disagreement_path =
      absl::GetFlag(FLAGS_disagreement_csv_path);
  if (!disagreement_path.empty()) {
    CHECK(!std::filesystem::exists(disagreement_path))
        << "Refusing to overwrite " << disagreement_path;
    disagreement_csv.open(disagreement_path);
    CHECK(disagreement_csv.is_open()) << "Cannot open " << disagreement_path;
    disagreement_csv << std::setprecision(17)
                     << "context_id,camera_a,camera_b,time_a_s,time_b_s,"
                        "translation_m,yaw_deg\n";
    solver_node->RegisterSolutionCallback(
        [&](const control_loop::Context& context,
            const localization::UnambiguousSolution& solution) {
          for (size_t i = 0; i < solution.selected.size(); ++i) {
            for (size_t j = i + 1; j < solution.selected.size(); ++j) {
              const auto& a = solution.selected[i];
              const auto& b = solution.selected[j];
              const std::string camera_a(camera_names.at(a.input_index));
              const std::string camera_b(camera_names.at(b.input_index));
              const auto* image_a = context->GetMessage<camera::JpegBuffer>(
                  "second_bot_" + camera_a + "/jpeg_buffer");
              const auto* image_b = context->GetMessage<camera::JpegBuffer>(
                  "second_bot_" + camera_b + "/jpeg_buffer");
              CHECK(image_a != nullptr && image_b != nullptr);
              const auto& pose_a = a.estimate.pose;
              const auto& pose_b = b.estimate.pose;
              const double yaw_deg = std::abs(std::remainder(
                  pose_a.Rotation().Z().value() -
                      pose_b.Rotation().Z().value(),
                  2.0 * std::numbers::pi)) * 180.0 / std::numbers::pi;
              disagreement_csv
                  << context->id << ',' << camera_a << ',' << camera_b << ','
                  << image_a->timestamp << ',' << image_b->timestamp << ','
                  << pose_a.Translation().Distance(pose_b.Translation()).value()
                  << ',' << yaw_deg << '\n';
            }
          }
          CHECK(disagreement_csv.good()) << "Cannot write disagreement CSV";
        });
  }
  solver_node->SetRejectFarTags(absl::GetFlag(FLAGS_reject_far_tags));
  solver_node->RegisterCallback(
      [](const control_loop::Context& context) -> void {
        auto pose =
            context->GetMessage<localization::PositionEstimateMessage>("pose");
        if (pose != nullptr) {
          LOG(INFO) << *pose;
        }
      });
  control_loop.RegisterNode(solver_node);

  for (std::size_t i = 0; i < camera_names.size(); ++i) {
    const std::string name(camera_names[i]);
    const std::string prefix = "second_bot_" + name;
    const std::string jpeg_channel = prefix + "/jpeg_buffer";
    const std::string decoded_channel = prefix + "/hardware_decoded_image";
    const std::string detections_channel =
        prefix + "/hardware_apriltag_detections";
    const std::string config_path =
        (std::filesystem::path(absl::GetFlag(FLAGS_camera_config_directory)) /
         (name + "_camera.json")).string();

    auto disk_camera_node = std::make_shared<camera::UVCDiskCameraNode>(
        replay_paths[i], jpeg_channel, replay_offset);
    control_loop.RegisterDependencyNode(disk_camera_node);

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>(
            jpeg_channel, "/stream", 4971 + static_cast<int>(i));
    control_loop.RegisterNode(jpeg_buffer_streamer_node);

    auto gpu_decode_node = std::make_shared<camera::NvjpegDecodeNode>(
        jpeg_channel, decoded_channel, NVJPEG_OUTPUT_Y, thread_pool);
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

  auto simulation_position_sender_node =
      std::make_shared<simulation::SimulationPositionSenderNode>("pose");
  control_loop.RegisterNode(simulation_position_sender_node);

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();

  if (disagreement_csv.is_open()) {
    disagreement_csv.flush();
    CHECK(disagreement_csv.good()) << "Cannot flush disagreement CSV";
    disagreement_csv.close();
  }

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
