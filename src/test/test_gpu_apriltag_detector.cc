#include <atomic>
#include <memory>
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
#include "utils/stop.h"

ABSL_FLAG(std::string, config_path, "",          // NOLINT
          "path to the uvc config json file");   // NOLINT
ABSL_FLAG(int, max_frames, 100,                  // NOLINT
          "number of decoded frames to test");   // NOLINT
ABSL_FLAG(bool, log_empty_frames, false,         // NOLINT
          "log frames with no tag detections");  // NOLINT

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  stop::RegisterHandler();

  const std::string config_path = absl::GetFlag(FLAGS_config_path);
  CHECK(!config_path.empty()) << "--config_path is required";

  const int max_frames = absl::GetFlag(FLAGS_max_frames);
  CHECK(max_frames > 0) << "--max_frames must be greater than zero";

  camera::UVCCameraConfig config(config_path);

  auto uvc_camera_node = std::make_unique<camera::UVCCameraNode>(config);
  auto nvjpeg_decode_node = std::make_unique<camera::NvjpegDecodeNode>(
      "test_gpu_apriltag_detector", NVJPEG_OUTPUT_Y);
  std::string detector_config_path = config_path;
  auto detector_node = std::make_unique<apriltag::NvidiaApriltagDetectorNode>(
      config.width, config.height, detector_config_path);

  std::atomic<int> decoded_frames = 0;
  std::atomic<int> detection_frames = 0;
  std::atomic<int> submitted_frames = 0;
  std::atomic<int> tags_seen = 0;

  detector_node->RegisterCallback(
      [&detection_frames, &tags_seen](
          const std::shared_ptr<apriltag::NvidiaTagDetections>& detections)
          -> void {
        const int frame_index = ++detection_frames;
        tags_seen += static_cast<int>(detections->tag_detections.size());

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

  nvjpeg_decode_node->RegisterCallback(
      [detector = detector_node.get(), &decoded_frames, max_frames](
          const std::shared_ptr<camera::DecodedJpegBuffer>& buffer) -> void {
        detector->Detect(buffer);
        if (++decoded_frames >= max_frames) {
          stop::stop = true;
        }
      });

  uvc_camera_node->RegisterCallback(
      [decoder = nvjpeg_decode_node.get(), &submitted_frames,
       max_frames](const std::shared_ptr<camera::JpegBuffer>& buffer) -> void {
        if (submitted_frames.fetch_add(1) >= max_frames) {
          stop::stop = true;
          return;
        }
        decoder->Decode(buffer);
      });

  uvc_camera_node->Start();
  LOG(INFO) << "Running GPU AprilTag detector for " << max_frames
            << " decoded frames";

  stop::WaitUntilStop();

  uvc_camera_node.reset();
  nvjpeg_decode_node.reset();
  detector_node.reset();

  LOG(INFO) << "Finished GPU AprilTag detector test: decoded_frames="
            << decoded_frames.load()
            << " detection_frames=" << detection_frames.load()
            << " submitted_frames=" << submitted_frames.load()
            << " tags_seen=" << tags_seen.load();
  return 0;
}
