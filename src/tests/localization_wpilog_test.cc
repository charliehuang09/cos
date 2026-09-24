#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <frc/geometry/struct/Pose3dStruct.h>
#include <wpi/DataLogReader.h>
#include <wpi/MemoryBuffer.h>
#include <wpi/struct/Struct.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "control_loop/thread_pool.h"
#include "localization/position.h"
#include "localization/unambiguous_solver_node.h"
#include "localization/variance_calculator_node.h"
#include "logging/wpilog_writer.h"

using namespace std::chrono_literals;

ABSL_FLAG(std::string, log_path, "/cos-logs/second_bot/log102/left",
          "Directory of timestamped JPEG frames.");  // NOLINT
ABSL_FLAG(std::string, camera_config,
          "/root/constants/second_bot/left_camera.json",
          "Camera calibration JSON.");  // NOLINT
ABSL_FLAG(std::string, wpilog_path, "/root/tests/localization.wpilog",
          "Output WPILOG file.");  // NOLINT
ABSL_FLAG(int, timeout_seconds, 240,
          "Maximum time to wait for replay.");  // NOLINT

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  CHECK_GT(absl::GetFlag(FLAGS_timeout_seconds), 0);

  const std::string input_path = absl::GetFlag(FLAGS_log_path);
  const std::string config_path = absl::GetFlag(FLAGS_camera_config);
  const std::string output_path = absl::GetFlag(FLAGS_wpilog_path);
  std::atomic<size_t> detection_frames = 0;
  std::atomic<size_t> poses = 0;
  std::atomic<size_t> poses_with_variance = 0;

  control_loop::RioClock::EnableSimulation();
  {
    control_loop::ControlLoop control_loop(1ms);
    control_loop::ThreadPool thread_pool;
    control_loop.SetMaxContext(1);

    auto disk_camera = std::make_shared<camera::UVCDiskCameraNode>(
        input_path, "jpeg_buffer", camera::GetEarliestTimestamp(input_path));
    control_loop.RegisterDependancyNode(disk_camera);

    auto decode = std::make_shared<camera::NvjpegDecodeNode>(
        "jpeg_buffer", "decoded_image", NVJPEG_OUTPUT_Y, thread_pool);
    decode->EnableTiming("decoded_image:latency");
    control_loop.RegisterNode(decode);

    auto detector = std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
        "decoded_image", "tag_detections", config_path, thread_pool);
    detector->EnableTiming("tag_detections:latency");
    detector->RegisterCallback([&](const control_loop::Context& context) {
      if (context->GetMessage<apriltag::TagDetections>("tag_detections") !=
          nullptr) {
        ++detection_frames;
      }
    });
    control_loop.RegisterNode(detector);

    auto solver =
        std::make_shared<localization::UnambiguousSolverNode>("pose");
    solver->SetRejectFarTags(false);
    solver->AddCamera("tag_detections", camera::Intrinsics{config_path},
                      camera::Extrinsics{config_path}, control_loop);
    solver->RegisterCallback([&](const control_loop::Context& context) {
      if (context->GetMessage<localization::PositionEstimateMessage>("pose") !=
          nullptr) {
        ++poses;
      }
    });
    control_loop.RegisterNode(solver);

    auto variance = std::make_shared<localization::VarianceCalculatorNode>(
        "pose", "pose_with_variance");
    variance->RegisterCallback([&](const control_loop::Context& context) {
      if (context->GetMessage<localization::PositionEstimateMessage>(
              "pose_with_variance") != nullptr) {
        ++poses_with_variance;
      }
    });
    control_loop.RegisterNode(variance);

    auto wpilog = std::make_shared<logging::WPILogWriter>(output_path);
    control_loop.SetWPILogWriter(wpilog);
    control_loop.Start();
    const auto deadline =
        std::chrono::steady_clock::now() +
        std::chrono::seconds(absl::GetFlag(FLAGS_timeout_seconds));
    while (!disk_camera->IsPlaybackComplete()) {
      CHECK_LT(std::chrono::steady_clock::now(), deadline)
          << "Camera replay did not finish";
      std::this_thread::sleep_for(10ms);
    }
    control_loop.Stop();
    thread_pool.Shutdown();
    wpilog->Close();

    CHECK_GT(detection_frames.load(), 0U);
    CHECK_GT(poses.load(), 0U);
    CHECK_GT(poses_with_variance.load(), 0U);
    CHECK(std::filesystem::exists(output_path));
    CHECK_GT(std::filesystem::file_size(output_path), 128U);

    auto buffer = wpi::MemoryBuffer::GetFile(output_path);
    CHECK(buffer.has_value());
    wpi::log::DataLogReader reader(std::move(buffer.value()));
    CHECK(reader.IsValid());
    std::unordered_map<int, std::string> entry_names;
    std::unordered_map<std::string, std::string> entry_types;
    std::unordered_map<std::string, size_t> records;
    size_t nonempty_detections = 0;
    int64_t first_pose_timestamp = 0;
    int64_t last_pose_timestamp = 0;
    for (const auto& record : reader) {
      if (record.IsStart()) {
        wpi::log::StartRecordData start;
        CHECK(record.GetStartData(&start));
        entry_names[start.entry] = start.name;
        entry_types[std::string(start.name)] = start.type;
      } else if (!record.IsControl()) {
        const auto entry = entry_names.find(record.GetEntry());
        CHECK(entry != entry_names.end());
        ++records[entry->second];
        if (entry->second == "tag_detections") {
          constexpr size_t kDetectionBytes =
              wpi::GetStructSize<int32_t>() + 8 * wpi::GetStructSize<double>();
          CHECK_EQ(record.GetSize() % kDetectionBytes, 0U);
          if (record.GetSize() >= kDetectionBytes) {
            const auto bytes = record.GetRaw();
            CHECK_GT(wpi::UnpackStruct<int32_t>(bytes), 0);
            CHECK(std::isfinite(wpi::UnpackStruct<double>(
                bytes.subspan(wpi::GetStructSize<int32_t>()))));
            ++nonempty_detections;
          }
        }
        if (entry->second == "pose" || entry->second == "pose_with_variance") {
          CHECK_EQ(record.GetSize(), wpi::GetStructSize<frc::Pose3d>());
          const frc::Pose3d pose =
              wpi::UnpackStruct<frc::Pose3d>(record.GetRaw());
          CHECK(std::isfinite(pose.X().value()));
          CHECK(std::isfinite(pose.Y().value()));
          if (entry->second == "pose") {
            if (first_pose_timestamp == 0) {
              first_pose_timestamp = record.GetTimestamp();
            }
            last_pose_timestamp = record.GetTimestamp();
          }
        }
        if (entry->second == "pose/tag_ids" ||
            entry->second == "pose_with_variance/tag_ids") {
          std::vector<int64_t> values;
          CHECK(record.GetIntegerArray(&values));
          CHECK(!values.empty());
        }
        if (entry->second == "pose/distances" ||
            entry->second == "pose_with_variance/distances") {
          std::vector<double> values;
          CHECK(record.GetDoubleArray(&values));
          CHECK(!values.empty());
        }
        if (entry->second == "pose/variance" ||
            entry->second == "pose_with_variance/variance") {
          double value = 0.0;
          CHECK(record.GetDouble(&value));
          CHECK(std::isfinite(value));
        }
        if (entry->second == "tag_detections:multitag_solver/pos1" ||
            entry->second == "tag_detections:multitag_solver/pos2") {
          CHECK_EQ(record.GetSize() % wpi::GetStructSize<frc::Pose3d>(), 0U);
          if (record.GetSize() != 0) {
            const frc::Pose3d pose =
                wpi::UnpackStruct<frc::Pose3d>(record.GetRaw());
            CHECK(std::isfinite(pose.X().value()));
          }
        }
      }
    }
    CHECK_EQ(entry_types.at("tag_detections"),
             "struct:TagDetection[]");
    CHECK(!entry_types.contains("jpeg_buffer"));
    CHECK(!entry_types.contains("decoded_image"));
    CHECK_GT(records["decoded_image:latency"], 0U);
    CHECK_GT(records["tag_detections"], 0U);
    CHECK_GT(nonempty_detections, 0U);
    CHECK_GT(records["tag_detections:latency"], 0U);
    CHECK_GT(records["tag_detections:multitag_solver"], 0U);
    const std::string solver_channel = "tag_detections:multitag_solver";
    CHECK_EQ(entry_types.at(solver_channel + "/pos1"), "struct:Pose3d[]");
    CHECK_EQ(entry_types.at(solver_channel + "/pos2"), "struct:Pose3d[]");
    CHECK_EQ(entry_types.at(solver_channel + "/pos2_indices"), "int64[]");
    for (const std::string_view suffix : {"/pos1_variance", "/pos2_variance",
                                          "/pos1_distance", "/pos2_distance"}) {
      CHECK_EQ(entry_types.at(solver_channel + std::string(suffix)), "double[]");
      CHECK_EQ(records.at(solver_channel),
               records.at(solver_channel + std::string(suffix)));
    }
    CHECK_EQ(records.at(solver_channel), records.at(solver_channel + "/pos1"));
    CHECK_EQ(records.at(solver_channel), records.at(solver_channel + "/pos2"));
    CHECK_GT(records["pose"], 0U);
    CHECK_GT(records["pose_with_variance"], 0U);
    for (const std::string_view channel : {"pose", "pose_with_variance"}) {
      const std::string name(channel);
      CHECK_EQ(entry_types.at(name), "struct:Pose3d");
      CHECK_EQ(entry_types.at(name + "/tag_ids"), "int64[]");
      CHECK_EQ(entry_types.at(name + "/distances"), "double[]");
      CHECK_EQ(entry_types.at(name + "/variance"), "double");
      CHECK_EQ(records.at(name), records.at(name + "/tag_ids"));
      CHECK_EQ(records.at(name), records.at(name + "/distances"));
      CHECK_EQ(records.at(name), records.at(name + "/variance"));
    }
    CHECK_GT(last_pose_timestamp - first_pose_timestamp, 120'000'000)
        << "Replay ended before the source recording did";

    // The GPU integration tests exit after shutdown to avoid device teardown.
    std::_Exit(EXIT_SUCCESS);
  }
}
