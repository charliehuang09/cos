#include "localization/square_solver_node.h"
#include "absl/base/log_severity.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/jpeg_disk_camera.h"
#include "camera/nvjpeg_decode_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "localization/multi_tag_solver_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(bool, multi_tag_solve, false,                            // NOLINT
          "Use MultiTagSolverNode instead of SquareSolverNode.");  // NOLINT

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  control_loop::ControlLoop control_loop(1ms);
  control_loop::ThreadPool thread_pool;
  control_loop.SetMaxContext(1);
  control_loop.EnableLatencyLog();

  const std::string path = "/root/constants/dev-orin/camera.json";

  {
    auto jpeg_disk_camera_node = std::make_shared<camera::JpegDiskCamera>(
        "/cos-logs/log60/left", "jpeg_buffer");
    control_loop.RegisterDependancyNode(jpeg_disk_camera_node);

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>("jpeg_buffer",
                                                           "stream", 4971);
    control_loop.RegisterNode(jpeg_buffer_streamer_node);

    auto gpu_decode_node = std::make_shared<camera::NvjpegDecodeNode>(
        "jpeg_buffer", "gpu_decoded_image", NVJPEG_OUTPUT_Y, thread_pool);
    control_loop.RegisterNode(gpu_decode_node);
    gpu_decode_node->EnableTiming("gpu_decoded_image:latency");

    auto gpu_apriltag_detector_node =
        std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
            "gpu_decoded_image", "gpu_apriltag_detections", path, thread_pool);
    control_loop.RegisterNode(gpu_apriltag_detector_node);
    gpu_apriltag_detector_node->EnableTiming("gpu_apriltag_detections:latency");

    std::shared_ptr<control_loop::INode> solver_node;
    if (absl::GetFlag(FLAGS_multi_tag_solve)) {
      solver_node = std::make_shared<localization::MultiTagSolverNode>(
          "gpu_apriltag_detections", "pose", camera::Intrinsics{path},
          camera::Extrinsics{path});
    } else {
      solver_node = std::make_shared<localization::SquareSolverNode>(
          "gpu_apriltag_detections", "pose", camera::Intrinsics{path},
          camera::Extrinsics{path});
    }
    control_loop.RegisterNode(solver_node);
    solver_node->RegisterCallback(
        [](const control_loop::Context& context) -> void {
          auto pose =
              context->GetMessage<localization::AmbiguousEstimateMessage>(
                  "pose");
          if (pose != nullptr) {
            for (const auto& estimate : pose->estimates) {
              LOG(INFO) << estimate.pos1;
            }
          }
        });
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
