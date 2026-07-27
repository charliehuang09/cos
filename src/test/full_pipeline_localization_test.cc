#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "apriltag/cpu_apriltag_detector_node.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/cpu_decode_node.h"
#include "camera/jpeg_disk_camera.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "localization/joint_solver_node.h"
#include "utils/json.h"
#include "wpi/DataLogWriter.h"

namespace fs = std::filesystem;

ABSL_FLAG(std::string, image_folder, "bos-logs/log60",  // NOLINT
          "Root directory containing one encoded-JPEG directory per camera");
ABSL_FLAG(std::string, camera_manifest,
          "src/test/test_constants/camera_constants.json",  // NOLINT
          "JSON camera constants containing real intrinsics and extrinsics");
ABSL_FLAG(std::string, detector_config_path,
          "constants/dev-orin/camera.json",  // NOLINT
          "Fallback detector config for manifests that omit one");
ABSL_FLAG(std::string, output_wpilog,
          "logs/full_pipeline_localization.wpilog",  // NOLINT
          "Output WPILOG path");
ABSL_FLAG(std::string, decode_backend, "nvjpeg",  // NOLINT
          "JPEG decode backend: cpu or nvjpeg");
ABSL_FLAG(std::string, detector_backend, "nvidia",  // NOLINT
          "AprilTag detector backend: cpu or nvidia");
ABSL_FLAG(int, control_loop_period_ms, 1,  // NOLINT
          "Control-loop period used while replaying frames");
ABSL_FLAG(double, max_acceleration, 100.0,  // NOLINT
          "Acceleration above this value is logged as suspicious");
ABSL_FLAG(double, max_pose_jump, 1.0,  // NOLINT
          "Pose translation jump above this value is logged as suspicious");

namespace {

struct CameraMetrics {
  std::atomic<size_t> encoded_frames = 0;
  std::atomic<size_t> decoded_frames = 0;
  std::atomic<size_t> detection_batches = 0;
  std::atomic<size_t> tag_detections = 0;
};

struct PoseMetrics {
  std::atomic<size_t> published_poses = 0;
  std::atomic<size_t> invalid_poses = 0;
  std::atomic<size_t> empty_tag_poses = 0;
  std::atomic<size_t> suspicious_accelerations = 0;
  std::atomic<size_t> suspicious_jumps = 0;
  std::mutex mutex;
  bool has_previous = false;
  double previous_timestamp = 0.0;
  std::array<double, 3> previous_translation = {};
  std::array<double, 3> previous_velocity = {};
};

enum class DecodeBackend { kCpu, kNvjpeg };
enum class DetectorBackend { kCpu, kNvidia };

auto ParseDecodeBackend(std::string_view value) -> DecodeBackend {
  if (value == "cpu") {
    return DecodeBackend::kCpu;
  }
  if (value == "nvjpeg") {
    return DecodeBackend::kNvjpeg;
  }
  LOG(FATAL) << "Unknown --decode_backend: " << value;
}

auto ParseDetectorBackend(std::string_view value) -> DetectorBackend {
  if (value == "cpu") {
    return DetectorBackend::kCpu;
  }
  if (value == "nvidia") {
    return DetectorBackend::kNvidia;
  }
  LOG(FATAL) << "Unknown --detector_backend: " << value;
}

auto ResolvePath(std::string_view raw, const fs::path& manifest_directory)
    -> fs::path {
  const fs::path supplied(raw);
  if (supplied.is_absolute()) {
    return fs::absolute(supplied);
  }
  return fs::absolute(manifest_directory / supplied);
}

auto IsJpeg(const fs::path& path) -> bool {
  std::string extension = path.extension().string();
  std::ranges::transform(extension, extension.begin(),
                         [](unsigned char character) -> char {
                           return static_cast<char>(std::tolower(character));
                         });
  return extension == ".jpg" || extension == ".jpeg";
}

auto HasFiniteTimestamp(const fs::path& path) -> bool {
  const std::string stem = path.stem().string();
  try {
    size_t parsed_characters = 0;
    const double timestamp = std::stod(stem, &parsed_characters);
    return parsed_characters == stem.size() && std::isfinite(timestamp);
  } catch (const std::invalid_argument&) {
    return false;
  } catch (const std::out_of_range&) {
    return false;
  }
}

auto CountReplayFrames(const fs::path& directory) -> size_t {
  CHECK(fs::is_directory(directory))
      << "JPEG directory does not exist: " << directory;
  size_t count = 0;
  for (const auto& entry : fs::directory_iterator(directory)) {
    if (entry.is_regular_file() && IsJpeg(entry.path()) &&
        HasFiniteTimestamp(entry.path())) {
      ++count;
    }
  }
  CHECK_GT(count, 0U) << "No timestamped JPEGs found in " << directory;
  return count;
}

auto CameraDirectoryName(std::string_view camera_name) -> std::string {
  const size_t separator = camera_name.rfind('_');
  if (separator == std::string_view::npos ||
      separator + 1 == camera_name.size()) {
    return std::string(camera_name);
  }
  return std::string(camera_name.substr(separator + 1));
}

auto CameraMatchesDirectory(const nlohmann::json& camera,
                            std::string_view directory_name) -> bool {
  const std::string name = camera.at("name").get<std::string>();
  if (name == directory_name || CameraDirectoryName(name) == directory_name) {
    return true;
  }
  return camera.contains("log_directory") &&
         camera.at("log_directory").get<std::string>() == directory_name;
}

auto LoadCameraConstants(const fs::path& manifest_path,
                         const fs::path& image_folder,
                         std::vector<fs::path>& image_directories)
    -> std::vector<localization::camera_constant_t> {
  const nlohmann::json manifest = utils::ReadJson(manifest_path.string());
  CHECK(manifest.contains("cameras"))
      << "Camera manifest has no cameras array: " << manifest_path;
  CHECK(manifest.at("cameras").is_array());
  CHECK(fs::is_directory(image_folder))
      << "Image folder does not exist: " << image_folder;

  for (const auto& entry : fs::directory_iterator(image_folder)) {
    if (entry.is_directory()) {
      image_directories.push_back(entry.path());
    }
  }
  std::ranges::sort(image_directories, {},
                    [](const fs::path& path) { return path.filename(); });
  CHECK(!image_directories.empty())
      << "Image folder contains no camera directories: " << image_folder;

  const fs::path manifest_directory = manifest_path.parent_path();
  std::vector<localization::camera_constant_t> camera_constants;
  camera_constants.reserve(image_directories.size());
  for (const fs::path& image_directory : image_directories) {
    const std::string directory_name = image_directory.filename().string();
    const nlohmann::json* camera = nullptr;
    for (const auto& candidate : manifest.at("cameras")) {
      if (CameraMatchesDirectory(candidate, directory_name)) {
        CHECK(camera == nullptr) << "Multiple cameras in " << manifest_path
                                 << " match image directory " << directory_name;
        camera = &candidate;
      }
    }
    CHECK(camera != nullptr) << "No camera named " << directory_name
                             << " found in " << manifest_path;

    localization::camera_constant_t camera_constant;
    camera_constant.name = camera->at("name").get<std::string>();
    camera_constant.intrinsics_path =
        ResolvePath(camera->at("intrinsics_path").get<std::string>(),
                    manifest_directory)
            .string();
    camera_constant.extrinsics_path =
        ResolvePath(camera->at("extrinsics_path").get<std::string>(),
                    manifest_directory)
            .string();
    camera_constant.detector_config_path =
        camera->contains("detector_config_path")
            ? ResolvePath(camera->at("detector_config_path").get<std::string>(),
                          manifest_directory)
                  .string()
            : fs::absolute(absl::GetFlag(FLAGS_detector_config_path)).string();

    CHECK(fs::exists(camera_constant.intrinsics_path))
        << "Missing intrinsics for " << camera_constant.name << ": "
        << camera_constant.intrinsics_path;
    CHECK(fs::exists(camera_constant.extrinsics_path))
        << "Missing extrinsics for " << camera_constant.name << ": "
        << camera_constant.extrinsics_path;
    CHECK(fs::exists(camera_constant.detector_config_path))
        << "Missing detector config for " << camera_constant.name << ": "
        << camera_constant.detector_config_path;
    CountReplayFrames(image_directory);
    camera_constants.push_back(std::move(camera_constant));
  }
  return camera_constants;
}

auto TimestampMicros(double timestamp) -> int64_t {
  if (!std::isfinite(timestamp) || timestamp <= 0.0) {
    return 0;
  }
  return static_cast<int64_t>(timestamp * 1'000'000.0);
}

auto PoseTimestamp(const control_loop::Context& context,
                   const std::vector<std::string>& decoded_channels) -> double {
  double timestamp = 0.0;
  for (const std::string& channel : decoded_channels) {
    const auto* cpu_image =
        context->GetMessage<camera::DecodedImageBuffer>(channel);
    if (cpu_image != nullptr) {
      timestamp = std::max(timestamp, cpu_image->timestamp);
      continue;
    }
    const auto* image = context->GetMessage<camera::DecodedJpegBuffer>(channel);
    if (image != nullptr) {
      timestamp = std::max(timestamp, image->timestamp);
    }
  }
  return timestamp;
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  CHECK_GT(absl::GetFlag(FLAGS_control_loop_period_ms), 0);
  CHECK_GT(absl::GetFlag(FLAGS_max_acceleration), 0.0);
  CHECK_GT(absl::GetFlag(FLAGS_max_pose_jump), 0.0);

  const fs::path manifest_path = absl::GetFlag(FLAGS_camera_manifest);
  const fs::path image_folder = absl::GetFlag(FLAGS_image_folder);
  const fs::path output_path = fs::absolute(absl::GetFlag(FLAGS_output_wpilog));
  const DecodeBackend decode_backend =
      ParseDecodeBackend(absl::GetFlag(FLAGS_decode_backend));
  const DetectorBackend detector_backend =
      ParseDetectorBackend(absl::GetFlag(FLAGS_detector_backend));
  std::vector<fs::path> image_directories;
  const std::vector<localization::camera_constant_t> camera_constants =
      LoadCameraConstants(manifest_path, image_folder, image_directories);

  if (output_path.has_parent_path()) {
    std::error_code error;
    fs::create_directories(output_path.parent_path(), error);
    CHECK(!error) << "Failed to create WPILOG directory: "
                  << output_path.parent_path() << ": " << error.message();
  }

  std::error_code wpilog_error;
  wpi::log::DataLogWriter wpilog(output_path.string(), wpilog_error,
                                 "cos full pipeline replay");
  CHECK(!wpilog_error) << "Failed to open WPILOG " << output_path << ": "
                       << wpilog_error.message();
  // StructLogEntry registers the canonical WPILib Pose3d schema and packs the
  // value using the SDK's own struct serialization implementation.
  wpi::log::StructLogEntry<frc::Pose3d> pose_entry(wpilog, "localization/pose");
  const int log_encoded_total = wpilog.Start("replay/encoded_frames", "int64");
  const int log_decoded_total = wpilog.Start("replay/decoded_frames", "int64");
  const int log_detection_total =
      wpilog.Start("replay/detection_batches", "int64");
  const int log_pose_count = wpilog.Start("localization/pose_count", "int64");
  const int log_pose_x = wpilog.Start("localization/pose/x", "double");
  const int log_pose_y = wpilog.Start("localization/pose/y", "double");
  const int log_pose_z = wpilog.Start("localization/pose/z", "double");
  const int log_pose_roll = wpilog.Start("localization/pose/roll", "double");
  const int log_pose_pitch = wpilog.Start("localization/pose/pitch", "double");
  const int log_pose_yaw = wpilog.Start("localization/pose/yaw", "double");
  const int log_pose_variance =
      wpilog.Start("localization/pose/variance", "double");
  const int log_pose_loss = wpilog.Start("localization/pose/loss", "double");
  const int log_pose_tags = wpilog.Start("localization/pose/num_tags", "int64");
  const int log_velocity = wpilog.Start("localization/velocity", "double");
  const int log_acceleration =
      wpilog.Start("localization/acceleration", "double");
  const int log_pose_jump = wpilog.Start("localization/pose_jump", "double");
  const int log_warning = wpilog.Start("localization/suspicious", "string");
  std::vector<int> log_tag_ids;
  log_tag_ids.reserve(camera_constants.size());
  for (const auto& camera_constant : camera_constants) {
    log_tag_ids.push_back(
        wpilog.Start("replay/" + camera_constant.name + "/tag_id", "int64"));
  }

  const size_t expected_frames = std::accumulate(
      image_directories.begin(), image_directories.end(), size_t{0},
      [](size_t total, const fs::path& image_directory) {
        return total + CountReplayFrames(image_directory);
      });
  std::atomic<size_t> detection_batches = 0;
  std::atomic<size_t> encoded_frames_total = 0;
  std::atomic<size_t> decoded_frames_total = 0;
  std::atomic<bool> replay_complete = false;
  std::mutex completion_mutex;
  std::condition_variable completion_cv;

  control_loop::ControlLoop control_loop(
      std::chrono::milliseconds(absl::GetFlag(FLAGS_control_loop_period_ms)));
  control_loop::ThreadPool thread_pool;

  std::vector<CameraMetrics> camera_metrics(camera_constants.size());
  std::vector<std::string> decoded_channels;
  decoded_channels.reserve(camera_constants.size());
  for (size_t camera_id = 0; camera_id < camera_constants.size(); ++camera_id) {
    const auto& camera_constant = camera_constants[camera_id];
    const std::string jpeg_channel = "jpeg/" + camera_constant.name;
    const std::string decoded_channel = "decoded/" + camera_constant.name;
    const std::string detection_channel =
        localization::DetectionBatchChannel(camera_constant.name);
    decoded_channels.push_back(decoded_channel);
    auto jpeg_camera = std::make_shared<camera::JpegDiskCamera>(
        image_directories[camera_id].string(), jpeg_channel, false, true);
    control_loop.RegisterDependancyNode(jpeg_camera);
    jpeg_camera->RegisterCallback([&, camera_id, jpeg_channel](
                                      const control_loop::Context& context) {
      const auto* jpeg = context->GetMessage<camera::JpegBuffer>(jpeg_channel);
      if (jpeg == nullptr || jpeg->ptr == nullptr) {
        return;
      }
      ++camera_metrics[camera_id].encoded_frames;
      const size_t encoded = ++encoded_frames_total;
      wpilog.AppendInteger(log_encoded_total, static_cast<int64_t>(encoded), 0);
    });

    std::shared_ptr<control_loop::INode> decoder;
    if (decode_backend == DecodeBackend::kCpu) {
      decoder = std::make_shared<camera::CpuJpegDecodeNode>(
          jpeg_channel, decoded_channel, thread_pool);
    } else {
      decoder = std::make_shared<camera::NvjpegDecodeNode>(
          jpeg_channel, decoded_channel, NVJPEG_OUTPUT_Y, thread_pool);
    }
    control_loop.RegisterNode(decoder);
    decoder->RegisterCallback(
        [&, camera_id, decoded_channel](const control_loop::Context& context) {
          const auto* cpu_image =
              context->GetMessage<camera::DecodedImageBuffer>(decoded_channel);
          const auto* image =
              context->GetMessage<camera::DecodedJpegBuffer>(decoded_channel);
          if (cpu_image == nullptr && image == nullptr) {
            return;
          }
          ++camera_metrics[camera_id].decoded_frames;
          const size_t decoded = ++decoded_frames_total;
          const double timestamp =
              cpu_image != nullptr ? cpu_image->timestamp : image->timestamp;
          wpilog.AppendInteger(log_decoded_total, static_cast<int64_t>(decoded),
                               TimestampMicros(timestamp));
        });

    std::shared_ptr<control_loop::INode> detector;
    if (detector_backend == DetectorBackend::kCpu) {
      detector = std::make_shared<apriltag::CpuApriltagDetectorNode>(
          decoded_channel, detection_channel,
          camera_constant.detector_config_path, thread_pool);
    } else {
      auto nvidia_detector =
          std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
              decoded_channel, detection_channel,
              camera_constant.detector_config_path, thread_pool);
      nvidia_detector->WarmUp();
      detector = std::move(nvidia_detector);
    }
    control_loop.RegisterNode(detector);
    detector->RegisterCallback([&, camera_id, detection_channel,
                                decoded_channel](
                                   const control_loop::Context& context) {
      const auto* detections =
          context->GetMessage<apriltag::TagDetections>(detection_channel);
      if (detections == nullptr) {
        return;
      }
      const auto* cpu_image =
          context->GetMessage<camera::DecodedImageBuffer>(decoded_channel);
      const auto* image =
          context->GetMessage<camera::DecodedJpegBuffer>(decoded_channel);
      const double image_timestamp = cpu_image != nullptr ? cpu_image->timestamp
                                     : image == nullptr   ? 0.0
                                                          : image->timestamp;
      const int64_t timestamp = TimestampMicros(image_timestamp);
      ++camera_metrics[camera_id].detection_batches;
      camera_metrics[camera_id].tag_detections +=
          detections->tag_detections.size();
      const size_t batches = ++detection_batches;
      wpilog.AppendInteger(log_detection_total, static_cast<int64_t>(batches),
                           timestamp);
      if (batches >= expected_frames) {
        // Camera logs may contain unequal frame counts. Signal from the
        // final detector batch instead of waiting for a final all-camera
        // solver rendezvous that cannot occur in that case.
        replay_complete.store(true);
        completion_cv.notify_one();
      }
      if (batches % 100 == 0 || batches == expected_frames) {
        const double progress = 100.0 * static_cast<double>(batches) /
                                static_cast<double>(expected_frames);
        LOG(INFO) << "Localization replay progress: " << batches << "/"
                  << expected_frames << " detection batches (" << progress
                  << "%)";
        if (batches % 100 == 0) {
          wpilog.Flush();
        }
      }
      for (const auto& detection : detections->tag_detections) {
        wpilog.AppendInteger(log_tag_ids[camera_id], detection.tag_id,
                             timestamp);
      }
    });
  }

  // The solver is owned by the control loop after registration. Registering an
  // observer through the node is the only way this test consumes a pose.
  auto solver_node = std::make_shared<localization::JointSolverNode>(
      "localization/position", camera_constants);
  control_loop.RegisterNode(solver_node);

  PoseMetrics pose_metrics;
  solver_node->RegisterCallback([&](const control_loop::Context& context) {
    const auto* estimate = context->GetMessage<localization::PositionEstimate>(
        "localization/position");
    if (estimate == nullptr) {
      return;
    }
    const double x = estimate->pose.X().value();
    const double y = estimate->pose.Y().value();
    const double z = estimate->pose.Z().value();
    const double roll = estimate->pose.Rotation().X().value();
    const double pitch = estimate->pose.Rotation().Y().value();
    const double yaw = estimate->pose.Rotation().Z().value();
    const int64_t timestamp =
        TimestampMicros(PoseTimestamp(context, decoded_channels));
    const size_t pose_count = ++pose_metrics.published_poses;
    wpilog.AppendInteger(log_pose_count, static_cast<int64_t>(pose_count),
                         timestamp);
    pose_entry.Append(estimate->pose, timestamp);
    wpilog.AppendDouble(log_pose_x, x, timestamp);
    wpilog.AppendDouble(log_pose_y, y, timestamp);
    wpilog.AppendDouble(log_pose_z, z, timestamp);
    wpilog.AppendDouble(log_pose_roll, roll, timestamp);
    wpilog.AppendDouble(log_pose_pitch, pitch, timestamp);
    wpilog.AppendDouble(log_pose_yaw, yaw, timestamp);
    wpilog.AppendDouble(log_pose_variance, estimate->variance, timestamp);
    wpilog.AppendDouble(log_pose_loss, estimate->loss, timestamp);
    wpilog.AppendInteger(log_pose_tags, estimate->num_tags, timestamp);

    if (estimate->invalid) {
      ++pose_metrics.invalid_poses;
      return;
    }
    if (estimate->tag_ids.empty()) {
      ++pose_metrics.empty_tag_poses;
      return;
    }

    std::lock_guard lock(pose_metrics.mutex);
    const std::array<double, 3> translation = {x, y, z};
    if (pose_metrics.has_previous && timestamp > 0 &&
        timestamp > TimestampMicros(pose_metrics.previous_timestamp)) {
      const double dt =
          timestamp / 1'000'000.0 - pose_metrics.previous_timestamp;
      const std::array<double, 3> velocity = {
          (x - pose_metrics.previous_translation[0]) / dt,
          (y - pose_metrics.previous_translation[1]) / dt,
          (z - pose_metrics.previous_translation[2]) / dt};
      const double speed =
          std::sqrt(velocity[0] * velocity[0] + velocity[1] * velocity[1] +
                    velocity[2] * velocity[2]);
      const double acceleration = std::sqrt(
          std::pow((velocity[0] - pose_metrics.previous_velocity[0]) / dt, 2) +
          std::pow((velocity[1] - pose_metrics.previous_velocity[1]) / dt, 2) +
          std::pow((velocity[2] - pose_metrics.previous_velocity[2]) / dt, 2));
      const double jump =
          std::sqrt(std::pow(x - pose_metrics.previous_translation[0], 2) +
                    std::pow(y - pose_metrics.previous_translation[1], 2) +
                    std::pow(z - pose_metrics.previous_translation[2], 2));
      wpilog.AppendDouble(log_velocity, speed, timestamp);
      wpilog.AppendDouble(log_acceleration, acceleration, timestamp);
      wpilog.AppendDouble(log_pose_jump, jump, timestamp);
      if (acceleration > absl::GetFlag(FLAGS_max_acceleration)) {
        ++pose_metrics.suspicious_accelerations;
        wpilog.AppendString(log_warning, "acceleration", timestamp);
      }
      if (jump > absl::GetFlag(FLAGS_max_pose_jump)) {
        ++pose_metrics.suspicious_jumps;
        wpilog.AppendString(log_warning, "pose_jump", timestamp);
      }
      pose_metrics.previous_velocity = velocity;
    }
    pose_metrics.previous_translation = translation;
    pose_metrics.previous_timestamp = timestamp / 1'000'000.0;
    pose_metrics.has_previous = true;
  });

  control_loop.Start();
  LOG(INFO) << "Running full localization replay for " << expected_frames
            << " JPEG frames across " << camera_constants.size() << " cameras";

  {
    std::unique_lock lock(completion_mutex);
    completion_cv.wait(
        lock, [&replay_complete] -> bool { return replay_complete.load(); });
  }

  control_loop.Stop();
  thread_pool.Shutdown();
  wpilog.Flush();

  const size_t encoded_frames = std::accumulate(
      camera_metrics.begin(), camera_metrics.end(), size_t{0},
      [](size_t total, const CameraMetrics& metrics) -> unsigned long {
        return total + metrics.encoded_frames.load();
      });
  const size_t decoded_frames = std::accumulate(
      camera_metrics.begin(), camera_metrics.end(), size_t{0},
      [](size_t total, const CameraMetrics& metrics) -> unsigned long {
        return total + metrics.decoded_frames.load();
      });
  CHECK_EQ(encoded_frames, expected_frames);
  CHECK_EQ(decoded_frames, expected_frames);
  CHECK_EQ(detection_batches.load(), expected_frames);
  CHECK_GT(pose_metrics.published_poses.load(), 0U)
      << "No PositionEstimate was published by the callback pipeline";

  const uintmax_t log_size = fs::file_size(output_path);
  CHECK_GE(log_size, 3U * 1024U)
      << "WPILOG is suspiciously small (" << log_size
      << " bytes); the pipeline likely wrote no data";

  LOG(INFO) << "Full replay complete: encoded=" << encoded_frames
            << " decoded=" << decoded_frames
            << " detection_batches=" << detection_batches.load()
            << " poses=" << pose_metrics.published_poses.load()
            << " suspicious_acceleration="
            << pose_metrics.suspicious_accelerations.load()
            << " suspicious_pose_jumps=" << pose_metrics.suspicious_jumps.load()
            << " wpilog_bytes=" << log_size;
  return EXIT_SUCCESS;
}
