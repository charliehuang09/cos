#include "apriltag/nvidia_apriltag_detector_node.h"
#include "absl/log/check.h"
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

  control_loop::ControlLoop control_loop(50ms);
  control_loop::ThreadPool thread_pool;

  std::atomic<int> total_tag_detections = 0;
  std::atomic<double> total_decode_latency = 0;
  std::atomic<size_t> total_decodes = 0;

  {
    auto jpeg_disk_camera_node = std::make_shared<camera::JpegDiskCamera>(
        "/cos-logs/log60/left", "jpeg_buffer");
    control_loop.RegisterDependancyNode(jpeg_disk_camera_node);

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>("jpeg_buffer",
                                                           "stream", 8080);
    control_loop.RegisterNode(jpeg_buffer_streamer_node);

    auto nvjpeg_decode_node = std::make_shared<camera::NvjpegDecodeNode>(
        "jpeg_buffer", "decoded_image", NVJPEG_OUTPUT_Y, thread_pool);
    control_loop.RegisterNode(nvjpeg_decode_node);
    nvjpeg_decode_node->EnableTiming("decoded_image:latency");
    nvjpeg_decode_node->RegisterCallback(
        [&total_decode_latency,
         &total_decodes](const control_loop::Context& context) -> void {
          auto detections = context->GetMessage<control_loop::LatencyMessage>(
              "decoded_image:latency");
          if (detections != nullptr) {
            total_decode_latency += detections->latency.count();
            total_decodes++;
          }
        });

    auto nvidia_apriltag_detector_node =
        std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
            "decoded_image", "apriltag_detections",
            "/root/constants/dev-orin/camera.json", thread_pool);
    control_loop.RegisterNode(nvidia_apriltag_detector_node);

    nvidia_apriltag_detector_node->RegisterCallback(
        [&total_tag_detections](const control_loop::Context& context) -> void {
          auto detections = context->GetMessage<apriltag::NvidiaTagDetections>(
              "apriltag_detections");
          if (detections == nullptr) {
            LOG(INFO) << -1;
            return;
          }
          LOG(INFO) << detections->tag_detections.size();
          total_tag_detections++;
        });
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();

  LOG(INFO) << "Total tag detections: " << total_tag_detections;
  LOG(INFO) << "Average decode latency: "
            << total_decode_latency / total_decodes;

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
