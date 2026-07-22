#include "apriltag/nvidia_apriltag_detector_node.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "camera/jpeg_disk_camera.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

auto main() -> int {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  // JPEG decode and PVA AprilTag detection take about 45 ms per frame on the
  // development Orin. Leave enough headroom for the asynchronous pipeline to
  // release its control-loop context before the next iteration.
  control_loop::ControlLoop control_loop(60ms);
  control_loop::ThreadPool thread_pool;

  auto jpeg_disk_camera_node = std::make_unique<camera::JpegDiskCamera>(
      "/cos-logs/log60/left", "jpeg_buffer");
  control_loop.RegisterDependancy(jpeg_disk_camera_node->CreateCallback());

  auto jpeg_buffer_streamer_node =
      std::make_unique<streamer::JpegBufferStreamerNode>("jpeg_buffer",
                                                         "stream", 8080);
  control_loop.RegisterCallback(jpeg_buffer_streamer_node->CreateCallback());

  auto nvjpeg_decode_node = std::make_unique<camera::NvjpegDecodeNode>(
      "jpeg_buffer", "decoded_image", NVJPEG_OUTPUT_Y, thread_pool);
  control_loop.RegisterCallback(nvjpeg_decode_node->CreateCallback());

  auto nvidia_apriltag_detector_node =
      std::make_unique<apriltag::NvidiaApriltagDetectorNode>(
          "decoded_image", "apriltag_detections",
          "/root/constants/dev-orin/camera.json", thread_pool);
  nvjpeg_decode_node->RegisterCallback(
      nvidia_apriltag_detector_node->CreateCallback());

  std::atomic<int> total_tag_detections = 0;
  nvidia_apriltag_detector_node->RegisterCallback(
      [&total_tag_detections](const control_loop::Context& context) -> void {
        auto detections = context->GetMessage<apriltag::NvidiaTagDetections>(
            "apriltag_detections");
        if (detections == nullptr) {
          return;
        }
        total_tag_detections++;
      });

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();

  LOG(INFO) << "Total tag detections: " << total_tag_detections;

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
