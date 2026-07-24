#include <filesystem>
#include <optional>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <wpi/system/Timer.hpp>

#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "control_loop/control_loop.h"
#include "logging/jpeg_buffer_log_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

#include "absl/log/globals.h"

ABSL_FLAG(std::string, config_path, "",         // NOLINT
          "path to the uvc config json file");  // NOLINT

ABSL_FLAG(                                                  // NOLINT
    std::optional<std::string>, stream_path, std::nullopt,  // NOLINT
    "Path for the stream. eg url is 10.9.71.101:8080/path. No stream if "  // NOLINT
    "left blank");  // NOLINT

ABSL_FLAG(std::optional<int>, port, std::nullopt,      // NOLINT
          "Streaming port. No stream if left blank");  // NOLINT

ABSL_FLAG(                                                        // NOLINT
    std::optional<std::string>, log_folder, std::nullopt,         // NOLINT
    "Folder for timestamped PNG frames. No logs if left blank");  // NOLINT

using namespace std::chrono_literals;

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  stop::RegisterHandler();

  camera::UVCCameraConfig config(absl::GetFlag(FLAGS_config_path));

  control_loop::ControlLoop control_loop;
  control_loop::ThreadPool thread_pool;

  auto uvc_camera_node =
      std::make_shared<camera::UVCCameraNode>("jpeg_stream", config);
  control_loop.RegisterDependancyNode(uvc_camera_node);

  auto nvjpeg_decode_node = std::make_shared<camera::NvjpegDecodeNode>(
      "jpeg_stream", "decoded_buffer", NVJPEG_OUTPUT_BGRI, thread_pool);
  control_loop.RegisterNode(nvjpeg_decode_node);

  if (absl::GetFlag(FLAGS_log_folder).has_value()) {
    const std::filesystem::path log_folder =
        absl::GetFlag(FLAGS_log_folder).value();
    CHECK(!log_folder.empty()) << "--log_folder must be empty";
    std::filesystem::create_directories(log_folder);

    auto jpeg_buffer_logger_node = std::make_shared<logging::JpegBufferLogNode>(
        "jpeg_stream", log_folder.string(), thread_pool);
    control_loop.RegisterNode(jpeg_buffer_logger_node);
  }

  if (absl::GetFlag(FLAGS_stream_path).has_value() &&
      absl::GetFlag(FLAGS_port).has_value()) {
    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>(
            "jpeg_stream", absl::GetFlag(FLAGS_stream_path).value(),
            absl::GetFlag(FLAGS_port).value());
    control_loop.RegisterNode(jpeg_buffer_streamer_node);
  }

  uvc_camera_node->Start();
  control_loop.Start();
  LOG(INFO) << "Started logging";

  stop::WaitUntilStop();

  thread_pool.Shutdown();
  control_loop.Stop();

  return 0;
}
