#include <atomic>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <string>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "control_loop/control_loop.h"

ABSL_FLAG(std::string, config_path, "",          // NOLINT
          "path to the uvc config json file");   // NOLINT
ABSL_FLAG(int, max_frames, 100,                  // NOLINT
          "number of decoded frames to test");   // NOLINT
ABSL_FLAG(bool, log_empty_frames, false,         // NOLINT
          "log frames with no tag detections");  // NOLINT

using namespace std::chrono_literals;

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const std::string config_path = absl::GetFlag(FLAGS_config_path);
  CHECK(!config_path.empty()) << "--config_path is required";

  const int max_frames = absl::GetFlag(FLAGS_max_frames);
  CHECK(max_frames > 0) << "--max_frames must be greater than zero";

  camera::UVCCameraConfig config(config_path);

  control_loop::ControlLoop control_loop(20ms);
  control_loop::ThreadPool thread_pool;

  auto uvc_camera_node =
      std::make_unique<camera::UVCCameraNode>("jpeg_stream", config);

  control_loop.RegisterDependancy(uvc_camera_node->CreateCallback());

  auto nvjpeg_decode_node = std::make_unique<camera::NvjpegDecodeNode>(
      "jpeg_stream", "decoded_image", NVJPEG_OUTPUT_Y, thread_pool);

  std::string detector_config_path = config_path;
  auto detector_node = std::make_unique<apriltag::NvidiaApriltagDetectorNode>(
      "decoded_image", "apriltag_detection", config.width, config.height,
      detector_config_path, thread_pool);

  std::atomic<int> decoded_frames = 0;
  std::atomic<int> detection_frames = 0;
  std::atomic<int> submitted_frames = 0;
  std::atomic<int> tags_seen = 0;
  std::mutex completion_mutex;
  std::condition_variable completion_cv;

  detector_node->RegisterCallback(
      [&detection_frames, &tags_seen, &completion_cv, max_frames](
          const std::shared_ptr<apriltag::NvidiaTagDetections>& detections)
          -> void {
        const int frame_index = ++detection_frames;
        tags_seen += static_cast<int>(detections->tag_detections.size());
        if (frame_index >= max_frames) {
          completion_cv.notify_one();
        }

        if (detections->tag_detections.empty()) {
          if (absl::GetFlag(FLAGS_log_empty_frames)) {
            LOG(INFO) << "frame=" << frame_index << " detections=0";
          }
          return;
        }

        for (const auto& detection : detections->tag_detections) {
          LOG(INFO) << "frame=" << frame_index << " tag_id=" << detection.tag_id
                    << " corners=[(" << detection.corners[0].x << ","
                    << detection.corners[0].y << "),(" << detection.corners[1].x
                    << "," << detection.corners[1].y << "),("
                    << detection.corners[2].x << "," << detection.corners[2].y
                    << "),(" << detection.corners[3].x << ","
                    << detection.corners[3].y << ")]";
        }
      });

  control_loop.RegisterCallback(nvjpeg_decode_node->CreateCallback());
  nvjpeg_decode_node->RegisterCallback(detector_node->CreateCallback());
  nvjpeg_decode_node->RegisterCallback(
      [&decoded_frames,
       &submitted_frames](const control_loop::Context& context) -> void {
        if (context->GetMessage<camera::DecodedJpegBuffer>("decoded_image") ==
            nullptr) {
          return;
        }
        if (submitted_frames.load() >= absl::GetFlag(FLAGS_max_frames)) {
          return;
        }
        ++decoded_frames;
        ++submitted_frames;
      });

  uvc_camera_node->Start();
  control_loop.Start();
  LOG(INFO) << "Running GPU AprilTag detector for " << max_frames
            << " decoded frames";

  {
    std::unique_lock lock(completion_mutex);
    completion_cv.wait(lock, [&detection_frames, max_frames]() -> bool {
      return detection_frames.load() >= max_frames;
    });
  }

  control_loop.Stop();
  thread_pool.Shutdown();
  uvc_camera_node.reset();
  nvjpeg_decode_node.reset();
  detector_node.reset();

  LOG(INFO) << "Finished GPU AprilTag detector test: decoded_frames="
            << decoded_frames.load()
            << " detection_frames=" << detection_frames.load()
            << " submitted_frames=" << submitted_frames.load()
            << " tags_seen=" << tags_seen.load();

  // Jetson's libnvidia-gpucomp process-exit finalizer double-frees memory when
  // VPI and nvJPEG are loaded together. All resources owned by this test have
  // been explicitly stopped and destroyed above, so bypass the broken library
  // finalizer.
  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
