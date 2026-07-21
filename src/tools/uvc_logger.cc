#include <filesystem>
#include <optional>
#include <string>
#include <vector>

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

ABSL_FLAG(std::optional<std::string>, log_folder, std::nullopt,      // NOLINT
          "Folder for timestamped PNG frames. No logs if left blank");  // NOLINT

using namespace std::chrono_literals;

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  stop::RegisterHandler();

  camera::UVCCameraConfig config(absl::GetFlag(FLAGS_config_path));

  control_loop::ControlLoop control_loop(20ms);
  control_loop::ThreadPool thread_pool;

  auto uvc_camera_node =
      std::make_unique<camera::UVCCameraNode>("jpeg_stream", config);
  control_loop.RegisterDependancy(uvc_camera_node->CreateCallback());

  auto nvjpeg_decode_node = std::make_unique<camera::NvjpegDecodeNode>(
      "jpeg_stream", "decoded_buffer", NVJPEG_OUTPUT_BGRI, thread_pool);
  control_loop.RegisterCallback(nvjpeg_decode_node->CreateCallback());

  if (absl::GetFlag(FLAGS_log_folder).has_value()) {
    const std::filesystem::path log_folder =
        absl::GetFlag(FLAGS_log_folder).value();
    CHECK(!log_folder.empty()) << "--log_folder must not be empty";
    std::filesystem::create_directories(log_folder);

    nvjpeg_decode_node->RegisterCallback(
        [log_folder](const control_loop::Context& context) -> void {
          auto* buffer =
              context->GetMessage<camera::DecodedJpegBuffer>("decoded_buffer");
          if (buffer == nullptr) {
            return;
          }
          CHECK(buffer->output_format == NVJPEG_OUTPUT_BGRI);

          std::vector<unsigned char> bgr(buffer->channel_sizes[0]);
          const cudaError_t status =
              cudaMemcpy(bgr.data(), buffer->destination.channel[0],
                         buffer->channel_sizes[0], cudaMemcpyDeviceToHost);
          CHECK(status == cudaSuccess) << cudaGetErrorString(status);

          const cv::Mat image(buffer->height, buffer->width, CV_8UC3,
                              bgr.data(), buffer->stride);
          const double timestamp = wpi::Timer::GetTimestamp().value();
          const std::filesystem::path image_path =
              log_folder / (std::to_string(timestamp) + ".png");
          CHECK(cv::imwrite(image_path.string(), image))
              << "Failed to write " << image_path;
        });
  }

  std::unique_ptr<streamer::JpegBufferStreamerNode> jpeg_buffer_streamer_node =
      nullptr;
  if (absl::GetFlag(FLAGS_stream_path).has_value() &&
      absl::GetFlag(FLAGS_port).has_value()) {
    jpeg_buffer_streamer_node =
        std::make_unique<streamer::JpegBufferStreamerNode>(
            "jpeg_stream", absl::GetFlag(FLAGS_stream_path).value(),
            absl::GetFlag(FLAGS_port).value());
    control_loop.RegisterCallback(jpeg_buffer_streamer_node->CreateCallback());
  }

  uvc_camera_node->Start();
  control_loop.Start();

  LOG(INFO) << "Started logging";

  stop::WaitUntilStop();

  control_loop.Stop();
  uvc_camera_node.reset();
  return 0;
}
