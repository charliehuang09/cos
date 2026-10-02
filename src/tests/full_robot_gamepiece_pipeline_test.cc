#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <unistd.h>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"
#include "apriltag/cpu_apriltag_detector_node.h"
#include "camera/jpeg_disk_camera.h"
#include "camera/cpu_decode_node.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/gamepiece_control_loop.h"
#include "gamepiece/gamepiece_detection.h"
#include "gamepiece/gamepiece_node.h"
#include "gamepiece/yolo.h"
#include "localization/position.h"
#include "localization/unambiguous_solver_node.h"

namespace fs = std::filesystem;
using namespace std::chrono_literals;

ABSL_FLAG(std::string, image_folder, "/root/gamepiece_logs/frames",  // NOLINT
          "Directory containing timestamped JPEG frames");
ABSL_FLAG(std::string, model_path, "/root/epoch80.engine",  // NOLINT
          "TensorRT engine with embedded NMS output");
ABSL_FLAG(std::string, annotation_dir, "/root/gamepiece_logs/annotated",  // NOLINT
          "Directory for frames annotated with TensorRT detections");
ABSL_FLAG(int, image_width, 640,  // NOLINT
          "Width of the replay JPEGs");
ABSL_FLAG(int, image_height, 360,  // NOLINT
          "Height of the replay JPEGs");
ABSL_FLAG(int, timeout_seconds, 300,  // NOLINT
          "Maximum time to wait for the replay to complete");

namespace {

auto IsTimestampedJpeg(const fs::path& path) -> bool {
  if (!path.has_extension()) {
    return false;
  }
  std::string extension = path.extension().string();
  std::ranges::transform(extension, extension.begin(),
                         [](unsigned char character) {
                           return static_cast<char>(std::tolower(character));
                         });
  if (extension != ".jpg" && extension != ".jpeg") {
    return false;
  }
  try {
    size_t parsed_characters = 0;
    const std::string stem = path.stem().string();
    std::stod(stem, &parsed_characters);
    return parsed_characters == stem.size();
  } catch (const std::exception&) {
    return false;
  }
}

auto CountFrames(const fs::path& directory) -> size_t {
  CHECK(fs::is_directory(directory)) << "Missing image folder: " << directory;
  const size_t count = std::ranges::count_if(
      fs::directory_iterator(directory), [](const fs::directory_entry& entry) {
        return entry.is_regular_file() && IsTimestampedJpeg(entry.path());
      });
  CHECK_GT(count, 0U) << "No timestamped JPEGs found in " << directory;
  return count;
}

auto DummyIntrinsics() -> nlohmann::json {
  return {{"cx", 640.0}, {"cy", 400.0}, {"fx", 905.0}, {"fy", 905.0},
          {"k1", 0.0},   {"k2", 0.0},   {"k3", 0.0},   {"p1", 0.0},
          {"p2", 0.0}};
}

auto DummyExtrinsics() -> nlohmann::json {
  return {{"translation_x", 0.0}, {"translation_y", 0.0},
          {"translation_z", 1.0}, {"rotation_x", 0.0},
          {"rotation_y", 0.35},   {"rotation_z", 0.0}};
}

class TemporaryCalibration final {
 public:
  TemporaryCalibration(int width, int height) {
    directory_ = fs::temp_directory_path() /
                 ("cos-gamepiece-calibration-" +
                  std::to_string(static_cast<long long>(getpid())));
    intrinsics_path_ = directory_ / "intrinsics.json";
    extrinsics_path_ = directory_ / "extrinsics.json";
    detector_config_path_ = directory_ / "detector.json";
    CHECK(fs::create_directory(directory_))
        << "Unable to create temporary calibration directory: "
        << directory_;
    WriteJson(intrinsics_path_, DummyIntrinsics());
    WriteJson(extrinsics_path_, DummyExtrinsics());
    WriteJson(detector_config_path_, { {"width", width}, {"height", height} });
  }

  ~TemporaryCalibration() {
    std::error_code error;
    fs::remove_all(directory_, error);
    if (error) {
      LOG(WARNING) << "Unable to remove temporary calibration directory: "
                   << directory_ << ": " << error.message();
    }
  }

  TemporaryCalibration(const TemporaryCalibration&) = delete;
  auto operator=(const TemporaryCalibration&) -> TemporaryCalibration& = delete;

  [[nodiscard]] auto IntrinsicsPath() const -> const fs::path& {
    return intrinsics_path_;
  }
  [[nodiscard]] auto ExtrinsicsPath() const -> const fs::path& {
    return extrinsics_path_;
  }
  [[nodiscard]] auto DetectorConfigPath() const -> const fs::path& {
    return detector_config_path_;
  }

 private:
  void WriteJson(const fs::path& path, const nlohmann::json& value) {
    std::ofstream stream(path);
    CHECK(stream.is_open()) << "Unable to write temporary calibration: "
                            << path;
    stream << value.dump(2) << '\n';
  }

  fs::path directory_;
  fs::path intrinsics_path_;
  fs::path extrinsics_path_;
  fs::path detector_config_path_;
};

class AnnotatingDetector final : public gamepiece::ObjectDetector {
 public:
  AnnotatingDetector(std::unique_ptr<gamepiece::ObjectDetector> detector,
                     fs::path output_directory)
      : detector_(std::move(detector)),
        output_directory_(std::move(output_directory)) {
    CHECK(detector_ != nullptr);
    CHECK(!fs::exists(output_directory_) || fs::is_empty(output_directory_))
        << "Annotation directory must be new or empty: " << output_directory_;
    fs::create_directories(output_directory_);
  }

  auto Detect(const cv::cuda::GpuMat& image)
      -> std::vector<gamepiece::LabeledBoundingBox> override {
    const std::vector<gamepiece::LabeledBoundingBox> detections =
        detector_->Detect(image);
    cv::Mat grayscale;
    image.download(grayscale);
    cv::Mat annotated;
    cv::cvtColor(grayscale, annotated, cv::COLOR_GRAY2BGR);
    for (const auto& detection : detections) {
      cv::rectangle(annotated, detection.bounds, cv::Scalar(0, 255, 0), 3);
      std::ostringstream label;
      label << "class_id=" << detection.class_id << " confidence="
            << std::fixed << std::setprecision(2) << detection.confidence;
      const cv::Point origin(detection.bounds.x,
                             std::max(24, detection.bounds.y - 8));
      cv::putText(annotated, label.str(), origin, cv::FONT_HERSHEY_SIMPLEX,
                  0.7, cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
    }
    std::lock_guard lock(output_mutex_);
    std::ostringstream filename;
    filename << std::setfill('0') << std::setw(6) << frame_index_++ << ".jpg";
    CHECK(cv::imwrite((output_directory_ / filename.str()).string(),
                      annotated));
    return detections;
  }

 private:
  std::unique_ptr<gamepiece::ObjectDetector> detector_;
  fs::path output_directory_;
  std::mutex output_mutex_;
  size_t frame_index_ = 0;
};

}  // namespace

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const fs::path image_folder = absl::GetFlag(FLAGS_image_folder);
  const fs::path model_path = absl::GetFlag(FLAGS_model_path);
  const fs::path annotation_dir = absl::GetFlag(FLAGS_annotation_dir);
  CHECK(fs::is_regular_file(model_path)) << "Missing engine: " << model_path;
  CHECK_GT(absl::GetFlag(FLAGS_image_width), 0);
  CHECK_GT(absl::GetFlag(FLAGS_image_height), 0);
  CHECK_GT(absl::GetFlag(FLAGS_timeout_seconds), 0);

  const size_t expected_frames = CountFrames(image_folder);
  TemporaryCalibration calibration(absl::GetFlag(FLAGS_image_width),
                                   absl::GetFlag(FLAGS_image_height));

  constexpr std::string_view kJpegChannel = "gamepiece/jpeg";
  constexpr std::string_view kLocalizationDecodedChannel =
      "localization/decoded";
  constexpr std::string_view kDecodedChannel = "gamepiece/decoded";
  constexpr std::string_view kAprilTagChannel =
      "localization/gamepiece_camera_detection_batch";
  constexpr std::string_view kPositionChannel = "localization/position";
  constexpr std::string_view kDetectionChannel = "gamepiece/detections";

  control_loop::ThreadPool thread_pool(1);
  control_loop::ControlLoop localization_loop(20ms);
  auto camera = std::make_shared<camera::JpegDiskCamera>(
      image_folder.string(), kJpegChannel, false, true);
  auto localization_decoder = std::make_shared<camera::CpuJpegDecodeNode>(
      kJpegChannel, kLocalizationDecodedChannel, thread_pool);
  auto decoder = std::make_shared<camera::NvjpegDecodeNode>(
      kJpegChannel, kDecodedChannel, NVJPEG_OUTPUT_Y, thread_pool);
  auto apriltag_detector =
      std::make_shared<apriltag::CpuApriltagDetectorNode>(
          kLocalizationDecodedChannel, kAprilTagChannel,
          calibration.DetectorConfigPath().string(),
          thread_pool);

  localization::camera_constant_t camera_constants{
      .name = "gamepiece_camera",
      .intrinsics_path = calibration.IntrinsicsPath().string(),
      .extrinsics_path = calibration.ExtrinsicsPath().string(),
      .detector_config_path = calibration.DetectorConfigPath().string(),
  };
  const std::vector<localization::camera_constant_t> camera_configs = {
      camera_constants};
  auto solver = std::make_shared<localization::UnambiguousSolverNode>(
      kPositionChannel, camera_configs);

  auto yolo = std::make_unique<gamepiece::Yolo>(model_path.string(),
                                                std::vector<std::string>{});
  auto annotating_detector = std::make_unique<AnnotatingDetector>(
      std::move(yolo), annotation_dir);
  auto gamepiece_node = std::make_shared<gamepiece::GamepieceNode>(
      std::move(annotating_detector), kDecodedChannel, kDetectionChannel,
      DummyIntrinsics(), DummyExtrinsics(), thread_pool);

  std::atomic<size_t> encoded_frames = 0;
  std::atomic<size_t> decoded_frames = 0;
  std::atomic<size_t> localization_detection_batches = 0;
  std::atomic<size_t> localization_callbacks = 0;
  std::atomic<size_t> gamepiece_batches = 0;
  std::atomic<size_t> total_detections = 0;
  std::atomic<size_t> handoff_frames = 0;
  std::atomic<size_t> unexpected_position_messages = 0;
  std::mutex completion_mutex;
  std::condition_variable completion;

  camera->RegisterCallback([&](const control_loop::Context& context) {
    const auto* jpeg = context->GetMessage<camera::JpegBuffer>(kJpegChannel);
    if (jpeg != nullptr && jpeg->ptr != nullptr) {
      ++encoded_frames;
    }
  });
  decoder->RegisterCallback([&](const control_loop::Context& context) {
    const auto* decoded =
        context->GetMessage<camera::DecodedJpegBuffer>(kDecodedChannel);
    if (decoded != nullptr && decoded->destination.channel[0] != nullptr) {
      ++decoded_frames;
    }
  });
  apriltag_detector->RegisterCallback(
      [&](const control_loop::Context& context) {
        const auto* detections =
            context->GetMessage<apriltag::TagDetections>(kAprilTagChannel);
        if (detections != nullptr) {
          ++localization_detection_batches;
        }
      });
  solver->RegisterCallback([&](const control_loop::Context& context) {
    ++localization_callbacks;
    (void)context;
  });
  gamepiece_node->RegisterCallback(
      [&](const control_loop::Context& context) {
        const auto* detections =
            context->GetMessage<gamepiece::GamepieceDetections>(
                kDetectionChannel);
        if (detections == nullptr) {
          return;
        }
        if (context->GetMessage<camera::DecodedJpegBuffer>(kDecodedChannel) !=
            nullptr) {
          ++handoff_frames;
        }
        if (context->GetMessage<localization::PositionEstimate>(
                kPositionChannel) != nullptr) {
          ++unexpected_position_messages;
        }
        total_detections += detections->detections.size();
        const size_t batches = ++gamepiece_batches;
        if (batches % 25 == 0 || batches == expected_frames) {
          LOG(INFO) << "Full robot replay progress: " << batches << "/"
                    << expected_frames << " gamepiece batches, detections="
                    << total_detections.load();
        }
        if (batches >= expected_frames) {
          completion.notify_one();
        }
      });

  gamepiece::GamepieceControlLoop gamepiece_loop(20ms);
  gamepiece_loop.RegisterDecodedFrameSource(decoder, kDecodedChannel);
  gamepiece_loop.RegisterNode(gamepiece_node);
  localization_loop.RegisterDependancyNode(camera);
  localization_loop.RegisterNode(localization_decoder);
  localization_loop.RegisterNode(decoder);
  localization_loop.RegisterNode(apriltag_detector);
  localization_loop.RegisterNode(solver);

  gamepiece_loop.Start();
  localization_loop.Start();
  LOG(INFO) << "Running full robot replay for " << expected_frames
            << " timestamped JPEG frames";

  bool completed = false;
  {
    std::unique_lock lock(completion_mutex);
    completed = completion.wait_for(
        lock, std::chrono::seconds(absl::GetFlag(FLAGS_timeout_seconds)), [&] {
          return gamepiece_batches.load() >= expected_frames;
        });
  }

  localization_loop.Stop();
  gamepiece_loop.Stop();
  thread_pool.Shutdown();

  CHECK(completed) << "Timed out after "
                   << absl::GetFlag(FLAGS_timeout_seconds)
                   << " seconds: gamepiece batches=" << gamepiece_batches.load()
                   << "/" << expected_frames;
  CHECK_EQ(encoded_frames.load(), expected_frames);
  CHECK_EQ(decoded_frames.load(), expected_frames);
  CHECK_EQ(localization_detection_batches.load(), expected_frames);
  CHECK_EQ(localization_callbacks.load(), expected_frames);
  CHECK_EQ(gamepiece_batches.load(), expected_frames);
  CHECK_EQ(handoff_frames.load(), expected_frames);
  CHECK_EQ(unexpected_position_messages.load(), 0U);
  CHECK_GT(total_detections.load(), 0U)
      << "The model produced no bounding boxes";

  size_t annotation_count = 0;
  for (const auto& entry : fs::directory_iterator(annotation_dir)) {
    if (entry.is_regular_file() && IsTimestampedJpeg(entry.path())) {
      ++annotation_count;
    }
  }
  CHECK_EQ(annotation_count, expected_frames);
  LOG(INFO) << "Full robot replay complete: encoded="
            << encoded_frames.load() << " decoded=" << decoded_frames.load()
            << " localization_batches="
            << localization_detection_batches.load()
            << " localization_callbacks=" << localization_callbacks.load()
            << " gamepiece_batches=" << gamepiece_batches.load()
            << " detections=" << total_detections.load()
            << " annotations=" << annotation_count;
  return 0;
}
