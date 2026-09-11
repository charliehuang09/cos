#include "localization/unambiguous_solver_node.h"
#include "absl/base/log_severity.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/get_earliest_timestamp.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "control_loop/thread_pool.h"
#include "simulation/simulation_position_sender_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(bool, reject_far_tags, true,                            // NOLINT
          "Reject tags and estimates that fail sanity checks.");  // NOLINT

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();
  control_loop::RioClock::EnableSimulation();

  control_loop::ControlLoop control_loop(1ms);
  control_loop::ThreadPool thread_pool;
  control_loop.SetMaxContext(1);
  control_loop.EnableLatencyLog();

  const std::string path = "/root/constants/second_bot/left_camera.json";
  const std::string log_path = "/cos-logs/second_bot/log102/left";

  {
    auto disk_camera_node = std::make_shared<camera::UVCDiskCameraNode>(
        log_path, "jpeg_buffer", camera::GetEarliestTimestamp(log_path));
    control_loop.RegisterDependancyNode(disk_camera_node);

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

    auto solver_node =
        std::make_shared<localization::UnambiguousSolverNode>("pose");
    solver_node->SetRejectFarTags(absl::GetFlag(FLAGS_reject_far_tags));
    solver_node->AddCamera("gpu_apriltag_detections", camera::Intrinsics{path},
                           camera::Extrinsics{path}, control_loop);
    solver_node->RegisterCallback(
        [](const control_loop::Context& context) -> void {
          auto pose =
              context->GetMessage<localization::PositionEstimateMessage>(
                  "pose");
          if (pose != nullptr) {
            LOG(INFO) << *pose;
          }
        });
    control_loop.RegisterNode(solver_node);

    auto simulation_position_sender_node =
        std::make_shared<simulation::SimulationPositionSenderNode>("pose");
    control_loop.RegisterNode(simulation_position_sender_node);
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();

  std::fflush(nullptr);
  std::_Exit(EXIT_SUCCESS);
}
