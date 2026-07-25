#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <numeric>
#include <limits>
#include <memory>
#include <mutex>
#include <numbers>
#include <span>
#include <string>
#include <string_view>
#include <stdexcept>
#include <vector>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/jpeg_disk_camera.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "localization/unambiguous_solver_node.h"
#include "utils/json.h"

namespace fs = std::filesystem;

ABSL_FLAG(std::string, log_path, "../bos/bos-logs/log181",  // NOLINT
          "Root directory containing one encoded-JPEG directory per camera");
ABSL_FLAG(std::string, camera_manifest, "../bos/constants/camera_constants.json",  // NOLINT
          "JSON camera manifest containing real intrinsics and extrinsics");
ABSL_FLAG(std::string, detector_config_path, "constants/dev-orin/camera.json",  // NOLINT
          "Camera JSON providing detector frame width and height");
ABSL_FLAG(std::string, output_wpilog, "full_pipeline_localization.wpilog",  // NOLINT
          "Output WPILOG path");
ABSL_FLAG(int, control_loop_period_ms, 1,  // NOLINT
          "Control-loop period used while replaying frames");
ABSL_FLAG(double, max_acceleration, 100.0,  // NOLINT
          "Acceleration above this value is logged as suspicious");
ABSL_FLAG(double, max_pose_jump, 1.0,  // NOLINT
          "Pose translation jump above this value is logged as suspicious");
ABSL_FLAG(int, timeout_seconds, 120,  // NOLINT
          "Maximum time to wait for all detector callbacks");

// The installed SDK exposes the datalog C ABI through libdatalog but does not
// install the corresponding C header. Keep the ABI declarations local to this
// executable rather than depending on SDK source-debug include paths.
extern "C" {
struct WPI_String {
  const char* str;
  size_t len;
};
struct WPI_DataLog;

WPI_DataLog* WPI_DataLog_CreateWriter(const WPI_String* filename,
                                      int* error_code,
                                      const WPI_String* extra_header);
void WPI_DataLog_Release(WPI_DataLog* datalog);
void WPI_DataLog_Flush(WPI_DataLog* datalog);
void WPI_DataLog_Stop(WPI_DataLog* datalog);
int WPI_DataLog_Start(WPI_DataLog* datalog, const WPI_String* name,
                      const WPI_String* type, const WPI_String* metadata,
                      int64_t timestamp);
void WPI_DataLog_AppendDouble(WPI_DataLog* datalog, int entry, double value,
                              int64_t timestamp);
void WPI_DataLog_AppendInteger(WPI_DataLog* datalog, int entry, int64_t value,
                               int64_t timestamp);
void WPI_DataLog_AppendString(WPI_DataLog* datalog, int entry,
                              const WPI_String* value, int64_t timestamp);
void WPI_DataLog_AppendRaw(WPI_DataLog* datalog, int entry,
                           const uint8_t* data, size_t len, int64_t timestamp);
void WPI_DataLog_AddSchemaString(WPI_DataLog* datalog,
                                 const WPI_String* name,
                                 const WPI_String* type,
                                 const WPI_String* schema,
                                 int64_t timestamp);
}

namespace {

auto String(std::string_view value) -> WPI_String {
  return WPI_String{value.data(), value.size()};
}

class WpiLog final {
 public:
  explicit WpiLog(const fs::path& path) {
    if (path.has_parent_path()) {
      std::error_code error;
      fs::create_directories(path.parent_path(), error);
      CHECK(!error) << "Failed to create WPILOG directory: "
                    << path.parent_path() << ": " << error.message();
    }

    const std::string filename = path.string();
    const WPI_String filename_string = String(filename);
    const WPI_String header = String("cos full pipeline replay");
    int error_code = 0;
    datalog_ = WPI_DataLog_CreateWriter(&filename_string, &error_code, &header);
    CHECK(datalog_ != nullptr) << "Failed to create WPILOG: " << filename;
    CHECK_EQ(error_code, 0) << "Failed to open WPILOG " << filename
                            << ", error=" << error_code;
  }

  ~WpiLog() {
    if (datalog_ != nullptr) {
      WPI_DataLog_Stop(datalog_);
      WPI_DataLog_Release(datalog_);
    }
  }

  WpiLog(const WpiLog&) = delete;
  auto operator=(const WpiLog&) -> WpiLog& = delete;

  auto Start(std::string_view name, std::string_view type) -> int {
    std::lock_guard lock(mutex_);
    const WPI_String name_string = String(name);
    const WPI_String type_string = String(type);
    const WPI_String metadata = String("");
    const int entry = WPI_DataLog_Start(datalog_, &name_string, &type_string,
                                        &metadata, 0);
    CHECK_GT(entry, 0) << "Failed to start WPILOG entry " << name;
    return entry;
  }

  void Double(int entry, double value, int64_t timestamp = 0) {
    std::lock_guard lock(mutex_);
    WPI_DataLog_AppendDouble(datalog_, entry, value, timestamp);
  }

  void Integer(int entry, int64_t value, int64_t timestamp = 0) {
    std::lock_guard lock(mutex_);
    WPI_DataLog_AppendInteger(datalog_, entry, value, timestamp);
  }

  void StringValue(int entry, std::string_view value, int64_t timestamp = 0) {
    std::lock_guard lock(mutex_);
    const WPI_String string = String(value);
    WPI_DataLog_AppendString(datalog_, entry, &string, timestamp);
  }

  void AddSchema(std::string_view name, std::string_view type,
                 std::string_view schema) {
    std::lock_guard lock(mutex_);
    const WPI_String name_string = String(name);
    const WPI_String type_string = String(type);
    const WPI_String schema_string = String(schema);
    WPI_DataLog_AddSchemaString(datalog_, &name_string, &type_string,
                                &schema_string, 0);
  }

  void Pose3d(int entry, const wpi::math::Pose3d& pose,
              int64_t timestamp = 0) {
    std::lock_guard lock(mutex_);
    std::array<uint8_t, 56> packed_pose{};
    wpi::util::Struct<wpi::math::Pose3d>::Pack(packed_pose, pose);
    WPI_DataLog_AppendRaw(datalog_, entry, packed_pose.data(), packed_pose.size(),
                          timestamp);
  }

  void Flush() {
    std::lock_guard lock(mutex_);
    WPI_DataLog_Flush(datalog_);
  }

 private:
  WPI_DataLog* datalog_ = nullptr;
  std::mutex mutex_;
};

struct CameraSpec {
  std::string name;
  fs::path image_path;
  fs::path intrinsics_path;
  fs::path extrinsics_path;
  fs::path detector_config_path;
  size_t expected_frames = 0;
};

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

auto ResolvePath(std::string_view raw, const fs::path& manifest_directory)
    -> fs::path {
  const fs::path supplied(raw);
  if (fs::exists(supplied)) {
    return supplied;
  }
  if (supplied.is_absolute()) {
    constexpr std::string_view kCosConstants = "/cos/constants/";
    if (raw.starts_with(kCosConstants)) {
      const fs::path relative(raw.substr(kCosConstants.size()));
      const fs::path fallback = manifest_directory / relative;
      if (fs::exists(fallback)) {
        return fallback;
      }
    }
    return supplied;
  }
  return manifest_directory / supplied;
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
  CHECK(fs::is_directory(directory)) << "JPEG directory does not exist: "
                                     << directory;
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
  if (separator == std::string_view::npos || separator + 1 == camera_name.size()) {
    return std::string(camera_name);
  }
  return std::string(camera_name.substr(separator + 1));
}

auto LoadCameraSpecs(const fs::path& manifest_path, const fs::path& log_path)
    -> std::vector<CameraSpec> {
  const nlohmann::json manifest = utils::ReadJson(manifest_path.string());
  CHECK(manifest.contains("cameras"))
      << "Camera manifest has no cameras array: " << manifest_path;
  CHECK(manifest.at("cameras").is_array());

  const fs::path manifest_directory = manifest_path.parent_path();
  std::vector<CameraSpec> cameras;
  for (const auto& camera : manifest.at("cameras")) {
    CameraSpec spec;
    spec.name = camera.at("name").get<std::string>();

    if (camera.contains("image_path")) {
      spec.image_path = ResolvePath(camera.at("image_path").get<std::string>(),
                                    manifest_directory);
    } else if (camera.contains("log_directory")) {
      spec.image_path = log_path / camera.at("log_directory").get<std::string>();
    } else {
      spec.image_path = log_path / CameraDirectoryName(spec.name);
    }

    spec.intrinsics_path = ResolvePath(
        camera.at("intrinsics_path").get<std::string>(), manifest_directory);
    spec.extrinsics_path = ResolvePath(
        camera.at("extrinsics_path").get<std::string>(), manifest_directory);
    if (camera.contains("detector_config_path")) {
      spec.detector_config_path = ResolvePath(
          camera.at("detector_config_path").get<std::string>(),
          manifest_directory);
    } else {
      spec.detector_config_path = ResolvePath(
          absl::GetFlag(FLAGS_detector_config_path), manifest_directory);
    }

    CHECK(fs::exists(spec.intrinsics_path))
        << "Missing intrinsics for " << spec.name << ": "
        << spec.intrinsics_path;
    CHECK(fs::exists(spec.extrinsics_path))
        << "Missing extrinsics for " << spec.name << ": "
        << spec.extrinsics_path;
    CHECK(fs::exists(spec.detector_config_path))
        << "Missing detector config for " << spec.name << ": "
        << spec.detector_config_path;
    spec.expected_frames = CountReplayFrames(spec.image_path);
    cameras.push_back(std::move(spec));
  }
  CHECK(!cameras.empty()) << "Camera manifest contains no cameras";
  return cameras;
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
    const auto* image =
        context->GetMessage<camera::DecodedJpegBuffer>(channel);
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
  CHECK_GT(absl::GetFlag(FLAGS_timeout_seconds), 0);

  const fs::path manifest_path = absl::GetFlag(FLAGS_camera_manifest);
  const fs::path log_path = absl::GetFlag(FLAGS_log_path);
  const fs::path output_path =
      fs::absolute(absl::GetFlag(FLAGS_output_wpilog));
  const std::vector<CameraSpec> cameras =
      LoadCameraSpecs(manifest_path, log_path);

  WpiLog wpilog(output_path);
  // This is the standard WPILib struct serialization for frc::Pose3d. The
  // SDK used by this project exposes the same type as wpi::math::Pose3d.
  wpilog.AddSchema("struct:Quaternion", "structschema",
                  "double w;double x;double y;double z");
  wpilog.AddSchema("struct:Translation3d", "structschema",
                  "double x;double y;double z");
  wpilog.AddSchema("struct:Rotation3d", "structschema", "Quaternion q");
  wpilog.AddSchema("struct:Pose3d", "structschema",
                  "Translation3d translation;Rotation3d rotation");
  const int log_encoded_total = wpilog.Start("replay/encoded_frames", "int64");
  const int log_decoded_total = wpilog.Start("replay/decoded_frames", "int64");
  const int log_detection_total =
      wpilog.Start("replay/detection_batches", "int64");
  const int log_pose_count = wpilog.Start("localization/pose_count", "int64");
  const int log_pose_struct =
      wpilog.Start("localization/pose", "struct:Pose3d");
  const int log_pose_x = wpilog.Start("localization/pose/x", "double");
  const int log_pose_y = wpilog.Start("localization/pose/y", "double");
  const int log_pose_z = wpilog.Start("localization/pose/z", "double");
  const int log_pose_roll =
      wpilog.Start("localization/pose/roll", "double");
  const int log_pose_pitch =
      wpilog.Start("localization/pose/pitch", "double");
  const int log_pose_yaw = wpilog.Start("localization/pose/yaw", "double");
  const int log_pose_variance =
      wpilog.Start("localization/pose/variance", "double");
  const int log_pose_loss = wpilog.Start("localization/pose/loss", "double");
  const int log_pose_tags = wpilog.Start("localization/pose/num_tags", "int64");
  const int log_velocity =
      wpilog.Start("localization/velocity", "double");
  const int log_acceleration =
      wpilog.Start("localization/acceleration", "double");
  const int log_pose_jump = wpilog.Start("localization/pose_jump", "double");
  const int log_warning = wpilog.Start("localization/suspicious", "string");
  std::vector<int> log_tag_ids;
  log_tag_ids.reserve(cameras.size());
  for (const CameraSpec& camera : cameras) {
    log_tag_ids.push_back(
        wpilog.Start("replay/" + camera.name + "/tag_id", "int64"));
  }

  const size_t expected_frames = std::accumulate(
      cameras.begin(), cameras.end(), size_t{0},
      [](size_t total, const CameraSpec& camera) {
        return total + camera.expected_frames;
      });
  std::atomic<size_t> detection_batches = 0;
  std::atomic<size_t> encoded_frames_total = 0;
  std::atomic<size_t> decoded_frames_total = 0;
  std::mutex completion_mutex;
  std::condition_variable completion_cv;

  control_loop::ControlLoop control_loop(std::chrono::milliseconds(
      absl::GetFlag(FLAGS_control_loop_period_ms)));
  control_loop::ThreadPool thread_pool;

  std::vector<CameraMetrics> camera_metrics(cameras.size());
  std::vector<std::string> decoded_channels;
  decoded_channels.reserve(cameras.size());
  std::vector<localization::camera_constant_t> solver_constants;
  solver_constants.reserve(cameras.size());

  for (size_t camera_id = 0; camera_id < cameras.size(); ++camera_id) {
    const CameraSpec& camera = cameras[camera_id];
    const std::string jpeg_channel = "jpeg/" + camera.name;
    const std::string decoded_channel = "decoded/" + camera.name;
    const std::string detection_channel =
        localization::DetectionBatchChannel(camera.name);
    decoded_channels.push_back(decoded_channel);
    solver_constants.push_back({camera.name, camera.intrinsics_path.string(),
                                camera.extrinsics_path.string()});

    auto jpeg_camera = std::make_shared<camera::JpegDiskCamera>(
        camera.image_path.string(), jpeg_channel, false, true);
    control_loop.RegisterDependancyNode(jpeg_camera);
    jpeg_camera->RegisterCallback(
        [&, camera_id, jpeg_channel](const control_loop::Context& context) {
          const auto* jpeg = context->GetMessage<camera::JpegBuffer>(jpeg_channel);
          if (jpeg == nullptr || jpeg->ptr == nullptr) {
            return;
          }
          ++camera_metrics[camera_id].encoded_frames;
          const size_t encoded = ++encoded_frames_total;
          wpilog.Integer(log_encoded_total, static_cast<int64_t>(encoded));
        });

    auto decoder = std::make_shared<camera::NvjpegDecodeNode>(
        jpeg_channel, decoded_channel, NVJPEG_OUTPUT_Y, thread_pool);
    control_loop.RegisterNode(decoder);
    decoder->RegisterCallback(
        [&, camera_id, decoded_channel](const control_loop::Context& context) {
          const auto* image =
              context->GetMessage<camera::DecodedJpegBuffer>(decoded_channel);
          if (image == nullptr) {
            return;
          }
          ++camera_metrics[camera_id].decoded_frames;
          const size_t decoded = ++decoded_frames_total;
          wpilog.Integer(log_decoded_total, static_cast<int64_t>(decoded),
                         TimestampMicros(image->timestamp));
        });

    auto detector = std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
        decoded_channel, detection_channel, camera.detector_config_path.string(),
        thread_pool);
    detector->WarmUp();
    control_loop.RegisterNode(detector);
    detector->RegisterCallback(
        [&, camera_id, detection_channel, decoded_channel](
            const control_loop::Context& context) {
          const auto* detections =
              context->GetMessage<apriltag::NvidiaTagDetections>(
                  detection_channel);
          if (detections == nullptr) {
            return;
          }
          const auto* image =
              context->GetMessage<camera::DecodedJpegBuffer>(decoded_channel);
          const int64_t timestamp =
              image == nullptr ? 0 : TimestampMicros(image->timestamp);
          ++camera_metrics[camera_id].detection_batches;
          camera_metrics[camera_id].tag_detections +=
              detections->tag_detections.size();
          const size_t batches = ++detection_batches;
          wpilog.Integer(log_detection_total, static_cast<int64_t>(batches),
                         timestamp);
          for (const auto& detection : detections->tag_detections) {
            wpilog.Integer(log_tag_ids[camera_id], detection.tag_id, timestamp);
          }

          if (batches >= expected_frames) {
            completion_cv.notify_one();
          }
        });
  }

  // The solver is owned by the control loop after registration. Registering an
  // observer through the node is the only way this test consumes a pose.
  auto solver_node = std::make_shared<localization::UnambiguousSolverNode>(
      "localization/position", solver_constants);
  control_loop.RegisterNode(solver_node);

  PoseMetrics pose_metrics;
  solver_node->RegisterCallback(
      [&](const control_loop::Context& context) {
        const auto* estimate =
            context->GetMessage<localization::PositionEstimate>(
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
        wpilog.Integer(log_pose_count, static_cast<int64_t>(pose_count),
                       timestamp);
        wpilog.Pose3d(log_pose_struct, estimate->pose, timestamp);
        wpilog.Double(log_pose_x, x, timestamp);
        wpilog.Double(log_pose_y, y, timestamp);
        wpilog.Double(log_pose_z, z, timestamp);
        wpilog.Double(log_pose_roll, roll, timestamp);
        wpilog.Double(log_pose_pitch, pitch, timestamp);
        wpilog.Double(log_pose_yaw, yaw, timestamp);
        wpilog.Double(log_pose_variance, estimate->variance, timestamp);
        wpilog.Double(log_pose_loss, estimate->loss, timestamp);
        wpilog.Integer(log_pose_tags, estimate->num_tags, timestamp);

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
          const double dt = timestamp / 1'000'000.0 -
                            pose_metrics.previous_timestamp;
          const std::array<double, 3> velocity = {
              (x - pose_metrics.previous_translation[0]) / dt,
              (y - pose_metrics.previous_translation[1]) / dt,
              (z - pose_metrics.previous_translation[2]) / dt};
          const double speed = std::sqrt(velocity[0] * velocity[0] +
                                         velocity[1] * velocity[1] +
                                         velocity[2] * velocity[2]);
          const double acceleration = std::sqrt(
              std::pow((velocity[0] - pose_metrics.previous_velocity[0]) / dt,
                       2) +
              std::pow((velocity[1] - pose_metrics.previous_velocity[1]) / dt,
                       2) +
              std::pow((velocity[2] - pose_metrics.previous_velocity[2]) / dt,
                       2));
          const double jump = std::sqrt(
              std::pow(x - pose_metrics.previous_translation[0], 2) +
              std::pow(y - pose_metrics.previous_translation[1], 2) +
              std::pow(z - pose_metrics.previous_translation[2], 2));
          wpilog.Double(log_velocity, speed, timestamp);
          wpilog.Double(log_acceleration, acceleration, timestamp);
          wpilog.Double(log_pose_jump, jump, timestamp);
          if (acceleration > absl::GetFlag(FLAGS_max_acceleration)) {
            ++pose_metrics.suspicious_accelerations;
            wpilog.StringValue(log_warning, "acceleration", timestamp);
          }
          if (jump > absl::GetFlag(FLAGS_max_pose_jump)) {
            ++pose_metrics.suspicious_jumps;
            wpilog.StringValue(log_warning, "pose_jump", timestamp);
          }
          pose_metrics.previous_velocity = velocity;
        }
        pose_metrics.previous_translation = translation;
        pose_metrics.previous_timestamp = timestamp / 1'000'000.0;
        pose_metrics.has_previous = true;
      });

  control_loop.Start();
  LOG(INFO) << "Running full localization replay for " << expected_frames
            << " JPEG frames across " << cameras.size() << " cameras";

  {
    std::unique_lock lock(completion_mutex);
    const bool completed = completion_cv.wait_for(
        lock, std::chrono::seconds(absl::GetFlag(FLAGS_timeout_seconds)),
        [&detection_batches, expected_frames] {
          return detection_batches.load() >= expected_frames;
        });
    CHECK(completed) << "Timed out waiting for detector callbacks: received "
                     << detection_batches.load() << " of " << expected_frames;
  }

  control_loop.Stop();
  thread_pool.Shutdown();
  wpilog.Flush();

  const size_t encoded_frames = std::accumulate(
      camera_metrics.begin(), camera_metrics.end(), size_t{0},
      [](size_t total, const CameraMetrics& metrics) {
        return total + metrics.encoded_frames.load();
      });
  const size_t decoded_frames = std::accumulate(
      camera_metrics.begin(), camera_metrics.end(), size_t{0},
      [](size_t total, const CameraMetrics& metrics) {
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
