#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <frc/geometry/struct/Pose3dStruct.h>
#include <nlohmann/json.hpp>
#include <wpi/DataLogReader.h>
#include <wpi/MemoryBuffer.h>
#include <wpi/struct/Struct.h>

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
#include "localization/unambiguous_solver_node.h"
#include "localization/variance_calculator_node.h"
#include "logging/wpilog_writer.h"
#include "simulation/simulation_position_sender_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(bool, reject_far_tags, true,                            // NOLINT
          "Reject tags and estimates that fail sanity checks.");  // NOLINT
ABSL_FLAG(std::string, log_path, "/cos-logs/second_bot/chezychamps",  // NOLINT
          "Directory containing one timestamped JPEG folder per camera");
ABSL_FLAG(std::string, camera_config_dir,                       // NOLINT
          "/root/constants/second_bot",                         // NOLINT
          "Directory containing camera calibration JSON files");
ABSL_FLAG(std::string, wpilog_path, "output_file.wpilog",  // NOLINT
          "Output WPILOG file");

namespace {

namespace fs = std::filesystem;

struct CameraReplay {
  fs::path image_dir;
  fs::path config_path;
  std::string name;
};

auto MatchesCamera(const std::string& directory_name,
                   const std::string& config_name,
                   const std::string& config_stem) -> bool {
  const auto matches = [&directory_name](const std::string& config) {
    constexpr std::string_view kCameraSuffix = "_camera";
    const std::string camera_name =
        config.ends_with(kCameraSuffix)
            ? config.substr(0, config.size() - kCameraSuffix.size())
            : config;
    return directory_name == config || directory_name == camera_name ||
           directory_name.ends_with("_" + camera_name) ||
           camera_name.ends_with("_" + directory_name);
  };
  return matches(config_name) || matches(config_stem);
}

auto FindCameraReplays(const fs::path& image_root, const fs::path& config_dir)
    -> std::vector<CameraReplay> {
  CHECK(fs::is_directory(image_root))
      << "Missing image directory: " << image_root;
  CHECK(fs::is_directory(config_dir))
      << "Missing camera constants: " << config_dir;

  struct CameraConfig {
    fs::path path;
    std::string name;
  };
  std::vector<CameraConfig> configs;
  for (const auto& entry : fs::directory_iterator(config_dir)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".json") {
      continue;
    }
    std::ifstream file(entry.path());
    CHECK(file.is_open()) << "Cannot read camera constants: " << entry.path();
    const auto json = nlohmann::json::parse(file);
    configs.push_back({entry.path(), json.at("name").get<std::string>()});
  }

  std::vector<fs::path> image_dirs;
  for (const auto& entry : fs::directory_iterator(image_root)) {
    if (entry.is_directory()) {
      image_dirs.push_back(entry.path());
    }
  }
  std::ranges::sort(image_dirs);
  CHECK(!image_dirs.empty()) << "No camera folders in " << image_root;

  std::vector<CameraReplay> replays;
  std::set<fs::path> used_configs;
  for (const auto& image_dir : image_dirs) {
    const std::string directory_name = image_dir.filename().string();
    const CameraConfig* match = nullptr;
    for (const auto& config : configs) {
      if (MatchesCamera(directory_name, config.name,
                        config.path.stem().string())) {
        CHECK(match == nullptr)
            << "Multiple camera constants match " << image_dir;
        match = &config;
      }
    }
    CHECK(match != nullptr) << "No camera constants match " << image_dir;
    CHECK(used_configs.insert(match->path).second)
        << "Camera constants used by multiple folders: " << match->path;
    replays.push_back({image_dir, match->path, directory_name});
  }
  return replays;
}

void ValidateWPILog(const std::string& path,
                    const std::vector<CameraReplay>& replays) {
  CHECK(fs::exists(path));
  CHECK_GT(fs::file_size(path), 128U);
  auto buffer = wpi::MemoryBuffer::GetFile(path);
  CHECK(buffer.has_value());
  wpi::log::DataLogReader reader(std::move(buffer.value()));
  CHECK(reader.IsValid());

  std::unordered_map<int, std::string> entry_names;
  std::unordered_map<std::string, std::string> entry_types;
  std::unordered_map<std::string, size_t> records;
  std::set<std::string> detection_channels;
  for (const auto& replay : replays) {
    detection_channels.insert("gpu_apriltag_detections:" + replay.name);
  }
  for (const auto& record : reader) {
    if (record.IsStart()) {
      wpi::log::StartRecordData start;
      CHECK(record.GetStartData(&start));
      entry_names[start.entry] = start.name;
      entry_types[std::string(start.name)] = start.type;
      continue;
    }
    if (record.IsControl()) {
      continue;
    }

    const auto entry = entry_names.find(record.GetEntry());
    CHECK(entry != entry_names.end());
    const std::string& channel = entry->second;
    ++records[channel];
    if (detection_channels.contains(channel)) {
      constexpr size_t kDetectionBytes =
          wpi::GetStructSize<int32_t>() + 8 * wpi::GetStructSize<double>();
      CHECK_EQ(record.GetSize() % kDetectionBytes, 0U);
      if (record.GetSize() != 0) {
        const auto bytes = record.GetRaw();
        CHECK_GT(wpi::UnpackStruct<int32_t>(bytes), 0);
        CHECK(std::isfinite(wpi::UnpackStruct<double>(
            bytes.subspan(wpi::GetStructSize<int32_t>()))));
      }
    } else if (channel == "pose" || channel == "pose_with_variance") {
      CHECK_EQ(record.GetSize(), wpi::GetStructSize<frc::Pose3d>());
      const frc::Pose3d pose = wpi::UnpackStruct<frc::Pose3d>(record.GetRaw());
      CHECK(std::isfinite(pose.X().value()));
      CHECK(std::isfinite(pose.Y().value()));
    } else if (channel == "pose/tag_ids" ||
               channel == "pose_with_variance/tag_ids") {
      std::vector<int64_t> values;
      CHECK(record.GetIntegerArray(&values));
      CHECK(!values.empty());
    } else if (channel == "pose/distances" ||
               channel == "pose_with_variance/distances") {
      std::vector<double> values;
      CHECK(record.GetDoubleArray(&values));
      CHECK(!values.empty());
    } else if (channel == "pose/variance" ||
               channel == "pose_with_variance/variance") {
      double value = 0.0;
      CHECK(record.GetDouble(&value));
      CHECK(std::isfinite(value));
    } else if (channel.ends_with(":multitag_solver/pos1") ||
               channel.ends_with(":multitag_solver/pos2")) {
      CHECK_EQ(record.GetSize() % wpi::GetStructSize<frc::Pose3d>(), 0U);
      if (record.GetSize() != 0) {
        const frc::Pose3d pose =
            wpi::UnpackStruct<frc::Pose3d>(record.GetRaw());
        CHECK(std::isfinite(pose.X().value()));
      }
    }
  }

  for (const auto& replay : replays) {
    const std::string jpeg = "jpeg_buffer:" + replay.name;
    const std::string decoded = "gpu_decoded_image:" + replay.name;
    const std::string detections = "gpu_apriltag_detections:" + replay.name;
    const std::string solver = detections + ":multitag_solver";
    CHECK(!entry_types.contains(jpeg));
    CHECK(!entry_types.contains(decoded));
    CHECK_EQ(entry_types.at(detections), "struct:TagDetection[]");
    CHECK_GT(records.at(decoded + ":latency"), 0U);
    CHECK_GT(records.at(detections), 0U);
    CHECK_GT(records.at(detections + ":latency"), 0U);
    CHECK_EQ(entry_types.at(solver), "string");
    CHECK_EQ(entry_types.at(solver + "/pos1"), "struct:Pose3d[]");
    CHECK_EQ(entry_types.at(solver + "/pos2"), "struct:Pose3d[]");
    CHECK_EQ(entry_types.at(solver + "/pos2_indices"), "int64[]");
    for (const std::string_view suffix : {"/pos1", "/pos2", "/pos2_indices",
                                          "/pos1_variance", "/pos2_variance",
                                          "/pos1_distance", "/pos2_distance"}) {
      const std::string field = solver + std::string(suffix);
      if (suffix != "/pos1" && suffix != "/pos2" &&
          suffix != "/pos2_indices") {
        CHECK_EQ(entry_types.at(field), "double[]");
      }
      CHECK_EQ(records[solver], records[field]);
    }
  }
  for (const std::string_view channel : {"pose", "pose_with_variance"}) {
    const std::string name(channel);
    CHECK_EQ(entry_types.at(name), "struct:Pose3d");
    CHECK_EQ(entry_types.at(name + "/tag_ids"), "int64[]");
    CHECK_EQ(entry_types.at(name + "/distances"), "double[]");
    CHECK_EQ(entry_types.at(name + "/variance"), "double");
    CHECK_EQ(records[name], records[name + "/tag_ids"]);
    CHECK_EQ(records[name], records[name + "/distances"]);
    CHECK_EQ(records[name], records[name + "/variance"]);
  }
}

}  // namespace

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();
  control_loop::RioClock::EnableSimulation();

  control_loop::ControlLoop control_loop(1ms);
  control_loop::ThreadPool thread_pool;
  control_loop.SetMaxContext(1);
  control_loop.EnableLatencyLog();
  auto wpilog = std::make_shared<logging::WPILogWriter>(
      absl::GetFlag(FLAGS_wpilog_path));
  control_loop.SetWPILogWriter(wpilog);

  const auto replays =
      FindCameraReplays(absl::GetFlag(FLAGS_log_path),
                        absl::GetFlag(FLAGS_camera_config_dir));
  std::vector<std::string> image_paths;
  for (const auto& replay : replays) {
    image_paths.push_back(replay.image_dir.string());
    LOG(INFO) << "Using " << replay.image_dir << " with " << replay.config_path;
  }
  const double first_timestamp = camera::GetEarliestTimestamp(image_paths);
  std::vector<std::shared_ptr<camera::UVCDiskCameraNode>> disk_cameras;

  {
    auto solver_node =
        std::make_shared<localization::UnambiguousSolverNode>("pose");
    solver_node->SetRejectFarTags(absl::GetFlag(FLAGS_reject_far_tags));

    int stream_port = 4971;
    for (const auto& replay : replays) {
      const std::string jpeg_channel = "jpeg_buffer:" + replay.name;
      const std::string decoded_channel = "gpu_decoded_image:" + replay.name;
      const std::string detections_channel =
          "gpu_apriltag_detections:" + replay.name;
      const std::string config_path = replay.config_path.string();

      auto disk_camera = std::make_shared<camera::UVCDiskCameraNode>(
          replay.image_dir.string(), jpeg_channel, first_timestamp,
          /*stop_on_complete=*/false, /*start_immediately=*/false);
      control_loop.RegisterDependancyNode(disk_camera);
      disk_cameras.push_back(disk_camera);

      auto jpeg_streamer = std::make_shared<streamer::JpegBufferStreamerNode>(
          jpeg_channel, "stream", stream_port++);
      control_loop.RegisterNode(jpeg_streamer);

      auto decoder = std::make_shared<camera::NvjpegDecodeNode>(
          jpeg_channel, decoded_channel, NVJPEG_OUTPUT_Y, thread_pool);
      control_loop.RegisterNode(decoder);
      decoder->EnableTiming(decoded_channel + ":latency");

      auto detector = std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
          decoded_channel, detections_channel, config_path, thread_pool);
      control_loop.RegisterNode(detector);
      detector->EnableTiming(detections_channel + ":latency");

      solver_node->AddCamera(detections_channel,
                             camera::Intrinsics{config_path},
                             camera::Extrinsics{config_path}, control_loop);
    }

    solver_node->RegisterCallback(
        [](const control_loop::Context& context) -> void {
          auto pose =
              context->GetMessage<localization::PositionEstimateMessage>(
                  "pose");
          if (pose != nullptr) {
            LOG(INFO) << *pose;
          }
        });
    control_loop.RegisterNode(solver_node);

    auto variance_node = std::make_shared<localization::VarianceCalculatorNode>(
        "pose", "pose_with_variance");
    control_loop.RegisterNode(variance_node);

    auto simulation_position_sender_node =
        std::make_shared<simulation::SimulationPositionSenderNode>("pose");
    control_loop.RegisterNode(simulation_position_sender_node);
  }

  control_loop::RioClock::Restart();
  control_loop.Start();
  for (const auto& camera : disk_cameras) {
    camera->StartPlayback();
  }

  while (!stop::StopRequested()) {
    const bool complete =
        std::ranges::all_of(disk_cameras, [](const auto& camera) {
          return camera->IsPlaybackComplete();
        });
    if (complete) {
      break;
    }
    std::this_thread::sleep_for(10ms);
  }

  control_loop.Stop();
  thread_pool.Shutdown();
  wpilog->Close();
  if (!stop::StopRequested()) {
    ValidateWPILog(absl::GetFlag(FLAGS_wpilog_path), replays);
  }

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
