#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "apriltag/nvidia_apriltag_detector_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "camera/uvc_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/thread_pool.h"
#include "localization/position_estimate_sender_node.h"
#include "localization/unambiguous_solver_node.h"
#include "networktables/NetworkTableInstance.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  stop::RegisterHandler();

  control_loop::ControlLoop control_loop(1ms);
  control_loop::ThreadPool thread_pool;
  const std::string path = "/root/constants/dev-orin/camera.json";

  auto uvc_camera_node = std::make_shared<camera::UVCCameraNode>(
      "jpeg_buffer",
      camera::UVCCameraConfig{"/root/constants/dev-orin/camera.json"});
  uvc_camera_node->Start();
  control_loop.RegisterDependancyNode(uvc_camera_node);

  auto jpeg_buffer_streamer_node =
      std::make_shared<streamer::JpegBufferStreamerNode>("jpeg_buffer",
                                                         "/stream", 4971);
  control_loop.RegisterNode(jpeg_buffer_streamer_node);

  auto hardware_decode_node = std::make_shared<camera::NvjpegFdDecodeNode>(
      "jpeg_buffer", "hardware_decoded_image", thread_pool);
  control_loop.RegisterNode(hardware_decode_node);
  hardware_decode_node->EnableTiming("hardware_decoded_image:latency");

  auto hardware_apriltag_detector_node =
      std::make_shared<apriltag::NvidiaApriltagDetectorNode>(
          "hardware_decoded_image", "hardware_apriltag_detections",
          "/root/constants/dev-orin/camera.json", thread_pool);
  control_loop.RegisterNode(hardware_apriltag_detector_node);
  hardware_apriltag_detector_node->EnableTiming(
      "hardware_apriltag_detections:latency");

  auto solver_node =
      std::make_shared<localization::UnambiguousSolverNode>("pose");
  solver_node->AddCamera("hardware_apriltag_detections",
                         camera::Intrinsics{path}, camera::Extrinsics{path},
                         control_loop);
  control_loop.RegisterNode(solver_node);

  auto networktables_instance = nt::NetworkTableInstance::Create();
  networktables_instance.StartServer();
  auto position_estimate_sender_node =
      std::make_shared<localization::PositionEstimateSenderNode>(
          "pose", "Orin/localization", networktables_instance);
  position_estimate_sender_node->SetLogEstimates(true);
  control_loop.RegisterNode(position_estimate_sender_node);

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
  thread_pool.Shutdown();
  networktables_instance.StopServer();
  nt::NetworkTableInstance::Destroy(networktables_instance);
}
