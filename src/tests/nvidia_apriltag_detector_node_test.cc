#include "apriltag/nvidia_apriltag_detector_node.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/gpu_apriltag_detector_lib.h"
#include "camera/jpeg_disk_camera.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "control_loop/timer.h"
#include "nvbufsurface.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

using namespace std::chrono_literals;

namespace {

struct DecoderMetrics {
  std::atomic<double> total_decode_latency = 0;
  std::atomic<size_t> total_decodes = 0;
  std::atomic<double> total_detection_latency = 0;
  std::atomic<size_t> total_detection_timings = 0;
  std::atomic<size_t> detection_frames = 0;
  std::atomic<size_t> total_tag_detections = 0;
};

}  // namespace

ABSL_FLAG(bool, gpu_decode, true, "");                              // NOLINT
ABSL_FLAG(bool, hardware_decode, true, "");                         // NOLINT
ABSL_FLAG(bool, new_gpu_detector, true,  // NOLINT
          "Run the new GPU detector with hardware JPEG decoding.");
ABSL_FLAG(uint, max_context, 1, "");                                // NOLINT
ABSL_FLAG(uint, instances, 1,                                       // NOLINT
          "Number of concurrent decode and detection pipelines.");  // NOLINT
ABSL_FLAG(std::optional<std::string>, log_path, std::nullopt, "");  // NOLINT

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  control_loop::ControlLoop control_loop(1ms);
  control_loop::ThreadPool thread_pool;
  control_loop.SetMaxContext(absl::GetFlag(FLAGS_max_context));
  control_loop.EnableLatencyLog();
  const uint instances = absl::GetFlag(FLAGS_instances);
  CHECK_GT(instances, 0U);

  DecoderMetrics gpu_metrics;
  DecoderMetrics hardware_metrics;
  DecoderMetrics new_gpu_metrics;

  {
    auto log_path = absl::GetFlag(FLAGS_log_path);
    if (log_path.has_value()) {
      auto jpeg_disk_camera_node = std::make_shared<camera::JpegDiskCamera>(
          log_path.value(), "jpeg_buffer");
      control_loop.RegisterDependancyNode(jpeg_disk_camera_node);
    } else {
      auto uvc_camera_node = std::make_shared<camera::UVCCameraNode>(
          "jpeg_buffer",
          camera::UVCCameraConfig{"/root/constants/dev-orin/camera.json"});
      uvc_camera_node->Start();
      control_loop.RegisterDependancyNode(uvc_camera_node);
    }

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>("jpeg_buffer",
                                                           "stream", 4971);
    control_loop.RegisterNode(jpeg_buffer_streamer_node);

    for (uint instance = 0; instance < instances; ++instance) {
      const std::string instance_suffix = "_" + std::to_string(instance);

      if (absl::GetFlag(FLAGS_gpu_decode)) {
        const std::string decoded_image_channel =
            "gpu_decoded_image" + instance_suffix;
        const std::string decode_latency_channel =
            decoded_image_channel + ":latency";
        const std::string detections_channel =
            "gpu_apriltag_detections" + instance_suffix;
        const std::string detection_latency_channel =
            detections_channel + ":latency";

        auto gpu_decode_node = std::make_shared<camera::NvjpegDecodeNode>(
            "jpeg_buffer", decoded_image_channel, NVJPEG_OUTPUT_Y, thread_pool);
        control_loop.RegisterNode(gpu_decode_node);
        gpu_decode_node->EnableTiming(decode_latency_channel);
        gpu_decode_node->RegisterCallback(
            [&gpu_metrics, decode_latency_channel](
                const control_loop::Context& context) -> void {
              auto latency = context->GetMessage<control_loop::LatencyMessage>(
                  decode_latency_channel);
              if (latency != nullptr) {
                gpu_metrics.total_decode_latency += latency->latency.count();
                gpu_metrics.total_decodes++;
              }
            });

        auto gpu_apriltag_detector_node =
            std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
                decoded_image_channel, detections_channel,
                "/root/constants/dev-orin/camera.json", thread_pool);
        control_loop.RegisterNode(gpu_apriltag_detector_node);
        gpu_apriltag_detector_node->EnableTiming(detection_latency_channel);
        gpu_apriltag_detector_node->RegisterCallback(
            [&gpu_metrics, detections_channel, detection_latency_channel](
                const control_loop::Context& context) -> void {
              auto latency = context->GetMessage<control_loop::LatencyMessage>(
                  detection_latency_channel);
              auto detections = context->GetMessage<apriltag::TagDetections>(
                  detections_channel);
              if (detections == nullptr) {
                return;
              }
              if (latency != nullptr) {
                gpu_metrics.total_detection_latency += latency->latency.count();
                gpu_metrics.total_detection_timings++;
              }
              gpu_metrics.detection_frames++;
              gpu_metrics.total_tag_detections +=
                  detections->tag_detections.size();
            });
      }

      if (absl::GetFlag(FLAGS_hardware_decode)) {
        const std::string decoded_image_channel =
            "hardware_decoded_image" + instance_suffix;
        const std::string decode_latency_channel =
            decoded_image_channel + ":latency";
        const std::string detections_channel =
            "hardware_apriltag_detections" + instance_suffix;
        const std::string detection_latency_channel =
            detections_channel + ":latency";

        auto hardware_decode_node =
            std::make_shared<camera::NvjpegFdDecodeNode>(
                "jpeg_buffer", decoded_image_channel, thread_pool);
        control_loop.RegisterNode(hardware_decode_node);
        hardware_decode_node->EnableTiming(decode_latency_channel);
        hardware_decode_node->RegisterCallback(
            [&hardware_metrics, decode_latency_channel](
                const control_loop::Context& context) -> void {
              auto latency = context->GetMessage<control_loop::LatencyMessage>(
                  decode_latency_channel);
              if (latency != nullptr) {
                hardware_metrics.total_decode_latency +=
                    latency->latency.count();
                hardware_metrics.total_decodes++;
              }
            });

        auto hardware_apriltag_detector_node =
            std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
                decoded_image_channel, detections_channel,
                "/root/constants/dev-orin/camera.json", thread_pool);
        control_loop.RegisterNode(hardware_apriltag_detector_node);
        hardware_apriltag_detector_node->EnableTiming(
            detection_latency_channel);

        hardware_apriltag_detector_node->RegisterCallback(
            [&hardware_metrics, detections_channel, detection_latency_channel](
                const control_loop::Context& context) -> void {
              auto detections = context->GetMessage<apriltag::TagDetections>(
                  detections_channel);
              if (detections == nullptr) {
                return;
              }
              auto latency = context->GetMessage<control_loop::LatencyMessage>(
                  detection_latency_channel);
              if (latency != nullptr) {
                hardware_metrics.total_detection_latency +=
                    latency->latency.count();
                hardware_metrics.total_detection_timings++;
              }
              hardware_metrics.detection_frames++;
              hardware_metrics.total_tag_detections +=
                  detections->tag_detections.size();
            });
      }

      if (absl::GetFlag(FLAGS_new_gpu_detector)) {
        const std::string decoded_image_channel =
            "new_gpu_decoded_image" + instance_suffix;
        const std::string decode_latency_channel =
            decoded_image_channel + ":latency";

        auto hardware_decode_node = std::make_shared<camera::NvjpegFdDecodeNode>(
            "jpeg_buffer", decoded_image_channel, thread_pool);
        control_loop.RegisterNode(hardware_decode_node);
        hardware_decode_node->EnableTiming(decode_latency_channel);

        const camera::UVCCameraConfig config{
            "/root/constants/dev-orin/camera.json"};
        const int width = config.width;
        const int height = config.height;
        auto detector =
            std::make_shared<apriltag::GpuApriltagDetector>(width, height);
        auto pixels = std::make_shared<std::vector<uint8_t>>(
            static_cast<size_t>(width) * height);
        auto detection_mutex = std::make_shared<std::mutex>();
        hardware_decode_node->RegisterCallback(
            [&new_gpu_metrics, decoded_image_channel, decode_latency_channel,
             detector, pixels, detection_mutex, width,
             height](const control_loop::Context& context) -> void {
              auto* buffer = context->GetMessage<camera::DecodedJpegFdBuffer>(
                  decoded_image_channel);
              if (buffer == nullptr) {
                return;
              }
              auto* latency = context->GetMessage<control_loop::LatencyMessage>(
                  decode_latency_channel);
              if (latency != nullptr) {
                new_gpu_metrics.total_decode_latency += latency->latency.count();
                new_gpu_metrics.total_decodes++;
              }

              // Include serialization and input preparation, as the VPI nodes do.
              control_loop::Timer timer;
              std::lock_guard lock(*detection_mutex);
              CHECK_EQ(buffer->width, width);
              CHECK_EQ(buffer->height, height);
              CHECK_GE(buffer->stride, static_cast<size_t>(width));
              NvBufSurface* surface = nullptr;
              CHECK_EQ(NvBufSurfaceFromFd(buffer->fd,
                                          reinterpret_cast<void**>(&surface)),
                       0);
              CHECK(surface != nullptr);
              CHECK_EQ(NvBufSurfaceMap(surface, 0, 0, NVBUF_MAP_READ), 0);
              CHECK_EQ(NvBufSurfaceSyncForCpu(surface, 0, 0), 0);
              const auto* data = static_cast<const uint8_t*>(
                  surface->surfaceList[0].mappedAddr.addr[0]);
              CHECK(data != nullptr);

              // The detector requires packed input; the decoder may pad rows.
              for (int row = 0; row < height; ++row) {
                std::memcpy(pixels->data() + static_cast<size_t>(row) * width,
                            data + static_cast<size_t>(row) * buffer->stride,
                            static_cast<size_t>(width));
              }
              CHECK_EQ(NvBufSurfaceUnMap(surface, 0, 0), 0);
              const apriltag::ImageView<uint8_t> view{
                  .data = pixels->data(),
                  .stride = width,
                  .height = height,
                  .width = width};
              const auto detections = detector->DetectAprilTag(view);
              new_gpu_metrics.total_detection_latency += timer.Stop().count();
              new_gpu_metrics.total_detection_timings++;
              new_gpu_metrics.detection_frames++;
              new_gpu_metrics.total_tag_detections += detections.size();
            });
      }
    }
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();

  const size_t gpu_decodes = gpu_metrics.total_decodes.load();
  const size_t hardware_decodes = hardware_metrics.total_decodes.load();
  const size_t gpu_detection_timings =
      gpu_metrics.total_detection_timings.load();
  const size_t hardware_detection_timings =
      hardware_metrics.total_detection_timings.load();
  if (absl::GetFlag(FLAGS_gpu_decode)) {
    CHECK_GT(gpu_decodes, 0U);
    CHECK_GT(gpu_detection_timings, 0U);
    LOG(INFO) << "GPU decode count: " << gpu_decodes;
    LOG(INFO) << "GPU detection frames: "
              << gpu_metrics.detection_frames.load();
    LOG(INFO) << "GPU tag detections: "
              << gpu_metrics.total_tag_detections.load();
    LOG(INFO) << "GPU average AprilTag detection latency: "
              << gpu_metrics.total_detection_latency / gpu_detection_timings;
    LOG(INFO) << "GPU average decode latency: "
              << gpu_metrics.total_decode_latency / gpu_decodes;
  }
  if (absl::GetFlag(FLAGS_hardware_decode)) {
    CHECK_GT(hardware_decodes, 0U);
    CHECK_GT(hardware_detection_timings, 0U);
    LOG(INFO) << "Hardware decode count: " << hardware_decodes;
    LOG(INFO) << "Hardware detection frames: "
              << hardware_metrics.detection_frames.load();
    LOG(INFO) << "Hardware tag detections: "
              << hardware_metrics.total_tag_detections.load();
    LOG(INFO) << "Hardware average AprilTag detection latency: "
              << hardware_metrics.total_detection_latency /
                     hardware_detection_timings;
    LOG(INFO) << "Hardware average decode latency: "
              << hardware_metrics.total_decode_latency / hardware_decodes;
  }

  if (absl::GetFlag(FLAGS_new_gpu_detector)) {
    const size_t decodes = new_gpu_metrics.total_decodes.load();
    const size_t detection_timings =
        new_gpu_metrics.total_detection_timings.load();
    CHECK_GT(decodes, 0U);
    CHECK_GT(detection_timings, 0U);
    LOG(INFO) << "New GPU decode count: " << decodes;
    LOG(INFO) << "New GPU detection frames: "
              << new_gpu_metrics.detection_frames.load();
    LOG(INFO) << "New GPU tag detections: "
              << new_gpu_metrics.total_tag_detections.load();
    LOG(INFO) << "New GPU average AprilTag detection latency: "
              << new_gpu_metrics.total_detection_latency / detection_timings;
    LOG(INFO) << "New GPU average decode latency: "
              << new_gpu_metrics.total_decode_latency / decodes;
  }

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
