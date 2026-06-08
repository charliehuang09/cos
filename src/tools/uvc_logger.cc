#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/initialize.h"

#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>

#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

#include "absl/log/globals.h"

ABSL_FLAG(int, fps, 0, "FPS");                                      // NOLINT
ABSL_FLAG(int, width, 0, "Width");                                  // NOLINT
ABSL_FLAG(int, height, 0, "Height");                                // NOLINT
ABSL_FLAG(std::string, serial_id, "", "Serial id");                 // NOLINT
ABSL_FLAG(std::string, log_folder, "", "Log folder (end with /)");  // NOLINT

ABSL_FLAG(std::string, path, "/stream",                             // NOLINT
          "Path for the stream. eg url is 10.9.71.101:8080/path");  // NOLINT
ABSL_FLAG(int, port, 5801,                                          // NOLINT
          "Streaming port");                                        // NOLINT
ABSL_FLAG(std::string, name, "UVC camera", "Name");                 // NOLINT
ABSL_FLAG(std::optional<int>, max_frame_size, std::nullopt,         // NOLINT
          "Max frame size");                                        // NOLINT
ABSL_FLAG(std::optional<int>, max_payload_size, std::nullopt,       // NOLINT
          "Max payload size");                                      // NOLINT

auto CopyPlane(const NvBuffer::NvBufferPlane& plane, int rows, int cols,
               unsigned char* dst) -> void {
  for (int y = 0; y < rows; ++y) {
    std::memcpy(dst + (y * cols), plane.data + (y * plane.fmt.stride), cols);
  }
}

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  stop::RegisterHandler();

  CHECK(absl::GetFlag(FLAGS_fps) != 0);
  CHECK(absl::GetFlag(FLAGS_width) != 0);
  CHECK(absl::GetFlag(FLAGS_height) != 0);
  CHECK(absl::GetFlag(FLAGS_serial_id) != "");
  CHECK(absl::GetFlag(FLAGS_log_folder) != "");

  camera::UVCCameraConfig config{
      .name = absl::GetFlag(FLAGS_name),
      .serial_id = absl::GetFlag(FLAGS_serial_id),
      .height = absl::GetFlag(FLAGS_height),
      .width = absl::GetFlag(FLAGS_width),
      .fps = absl::GetFlag(FLAGS_fps),
  };
  config.max_payload_size =
      absl::GetFlag(FLAGS_max_payload_size).value_or(config.max_payload_size);
  config.max_frame_size =
      absl::GetFlag(FLAGS_max_frame_size).value_or(config.max_frame_size);

  auto uvc_camera_node = std::make_unique<camera::UVCCameraNode>(config);
  auto jpeg_buffer_streamer_node =
      std::make_unique<streamer::JpegBufferStreamerNode>(
          absl::GetFlag(FLAGS_path), absl::GetFlag(FLAGS_port));

  auto nvjpeg_decode_node = std::make_unique<camera::NvjpegDecodeNode>("");

  uvc_camera_node->RegisterCallback(
      [streamer = jpeg_buffer_streamer_node.get()](const auto& buffer) {
        streamer->Stream(buffer);
      });

  uvc_camera_node->RegisterCallback(
      [decoder = nvjpeg_decode_node.get()](const auto& buffer) {
        decoder->Decode(buffer);
      });

  std::atomic<int> frame_index_atomic = 0;
  nvjpeg_decode_node->RegisterCallback(
      [frame_index_atomic = std::ref(frame_index_atomic),
       log_folder = absl::GetFlag(FLAGS_log_folder)](
          const std::shared_ptr<camera::DecodedJpegNvBuffer>& buffer) {
        const int height = buffer->buffer->planes[0].fmt.height;
        const int width = buffer->buffer->planes[0].fmt.width;

        cv::Mat i420(height + (height / 2), width, CV_8UC1);

        auto* y_dst = i420.ptr<unsigned char>(0);
        auto* u_dst = i420.ptr<unsigned char>(height);
        auto* v_dst = i420.ptr<unsigned char>(height + (height / 4));

        int frame_index = frame_index_atomic.get()++;

        CopyPlane(buffer->buffer->planes[0], static_cast<int>(height),
                  static_cast<int>(width), y_dst);
        CopyPlane(buffer->buffer->planes[1], static_cast<int>(height / 2),
                  static_cast<int>(width / 2), u_dst);
        CopyPlane(buffer->buffer->planes[2], static_cast<int>(height / 2),
                  static_cast<int>(width / 2), v_dst);

        cv::Mat out;
        cv::cvtColor(i420, out, cv::COLOR_YUV2BGR_I420);
        cv::imwrite(log_folder + std::to_string(frame_index) + ".png", out);
      });

  uvc_camera_node->Start();

  LOG(INFO) << "Started streaming";

  stop::WaitUntilStop();

  uvc_camera_node.reset();
}
