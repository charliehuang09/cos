#include "apriltag/nvidia_apriltag_detector_node.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "camera/jpeg_disk_camera.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

namespace {

struct DecoderMetrics {
  std::atomic<double> total_decode_latency = 0;
  std::atomic<size_t> total_decodes = 0;
  std::atomic<size_t> detection_frames = 0;
  std::atomic<size_t> total_tag_detections = 0;
};

}  // namespace

auto main() -> int {
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  control_loop::ControlLoop control_loop(50ms);
  control_loop::ThreadPool thread_pool;

  DecoderMetrics gpu_metrics;
  DecoderMetrics hardware_metrics;

  {
    auto jpeg_disk_camera_node = std::make_shared<camera::JpegDiskCamera>(
        "/cos-logs/log60/left", "jpeg_buffer");
    control_loop.RegisterDependancyNode(jpeg_disk_camera_node);

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>("jpeg_buffer",
                                                           "stream", 8080);
    control_loop.RegisterNode(jpeg_buffer_streamer_node);

    auto gpu_decode_node = std::make_shared<camera::NvjpegDecodeNode>(
        "jpeg_buffer", "gpu_decoded_image", NVJPEG_OUTPUT_Y, thread_pool);
    control_loop.RegisterNode(gpu_decode_node);
    gpu_decode_node->EnableTiming("gpu_decoded_image:latency");
    gpu_decode_node->RegisterCallback(
        [&gpu_metrics](const control_loop::Context& context) -> void {
          auto latency = context->GetMessage<control_loop::LatencyMessage>(
              "gpu_decoded_image:latency");
          if (latency != nullptr) {
            gpu_metrics.total_decode_latency += latency->latency.count();
            gpu_metrics.total_decodes++;
          }
        });

    auto hardware_decode_node =
        std::make_shared<camera::NvjpegFdDecodeNode>(
            "jpeg_buffer", "hardware_decoded_image", thread_pool);
    control_loop.RegisterNode(hardware_decode_node);
    hardware_decode_node->EnableTiming("hardware_decoded_image:latency");
    hardware_decode_node->RegisterCallback(
        [&hardware_metrics](const control_loop::Context& context) -> void {
          auto latency = context->GetMessage<control_loop::LatencyMessage>(
              "hardware_decoded_image:latency");
          if (latency != nullptr) {
            hardware_metrics.total_decode_latency += latency->latency.count();
            hardware_metrics.total_decodes++;
          }
        });

    auto gpu_apriltag_detector_node =
        std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
            "gpu_decoded_image", "gpu_apriltag_detections",
            "/root/constants/dev-orin/camera.json", thread_pool);
    control_loop.RegisterNode(gpu_apriltag_detector_node);

    gpu_apriltag_detector_node->RegisterCallback(
        [&gpu_metrics](const control_loop::Context& context) -> void {
          auto detections = context->GetMessage<apriltag::NvidiaTagDetections>(
              "gpu_apriltag_detections");
          if (detections == nullptr) {
            return;
          }
          gpu_metrics.detection_frames++;
          gpu_metrics.total_tag_detections +=
              detections->tag_detections.size();
        });

    auto hardware_apriltag_detector_node =
        std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
            "hardware_decoded_image", "hardware_apriltag_detections",
            "/root/constants/dev-orin/camera.json", thread_pool);
    control_loop.RegisterNode(hardware_apriltag_detector_node);

    hardware_apriltag_detector_node->RegisterCallback(
        [&hardware_metrics](const control_loop::Context& context) -> void {
          auto detections = context->GetMessage<apriltag::NvidiaTagDetections>(
              "hardware_apriltag_detections");
          if (detections == nullptr) {
            return;
          }
          hardware_metrics.detection_frames++;
          hardware_metrics.total_tag_detections +=
              detections->tag_detections.size();
        });
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();

  const size_t gpu_decodes = gpu_metrics.total_decodes.load();
  const size_t hardware_decodes = hardware_metrics.total_decodes.load();
  CHECK_GT(gpu_decodes, 0U);
  CHECK_GT(hardware_decodes, 0U);
  LOG(INFO) << "GPU decode count: " << gpu_decodes;
  LOG(INFO) << "GPU detection frames: "
            << gpu_metrics.detection_frames.load();
  LOG(INFO) << "GPU tag detections: "
            << gpu_metrics.total_tag_detections.load();
  LOG(INFO) << "GPU average decode latency: "
            << gpu_metrics.total_decode_latency / gpu_decodes;
  LOG(INFO) << "Hardware decode count: " << hardware_decodes;
  LOG(INFO) << "Hardware detection frames: "
            << hardware_metrics.detection_frames.load();
  LOG(INFO) << "Hardware tag detections: "
            << hardware_metrics.total_tag_detections.load();
  LOG(INFO) << "Hardware average decode latency: "
            << hardware_metrics.total_decode_latency / hardware_decodes;

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
