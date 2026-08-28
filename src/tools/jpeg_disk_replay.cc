#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"

#include "camera/get_earliest_timestamp.h"
#include "camera/uvc_disk_camera_node.h"
#include "control_loop/control_loop.h"
#include "control_loop/rio_clock.h"
#include "streamer/jpeg_buffer_streamer_node.h"
#include "utils/stop.h"

using namespace std::chrono_literals;

ABSL_FLAG(std::string, log_path, "",                                  // NOLINT
          "Folder for timestamped JPEG frames. No replay if blank");  // NOLINT

ABSL_FLAG(uint, port, 4971,  // NOLINT
          "Port");           // NOLINT

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);

  CHECK(absl::GetFlag(FLAGS_log_path) != "");
  stop::RegisterHandler();
  control_loop::RioClock::EnableSimulation();

  control_loop::ControlLoop control_loop(15ms);

  {

    const std::string log_path = absl::GetFlag(FLAGS_log_path);
    auto disk_camera_node = std::make_shared<camera::UVCDiskCameraNode>(
        log_path, "jpeg_stream", camera::GetEarliestTimestamp(log_path));

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>(
            "jpeg_stream", "/stream", absl::GetFlag(FLAGS_port));

    control_loop.RegisterDependancyNode(disk_camera_node);
    control_loop.RegisterNode(jpeg_buffer_streamer_node);
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
}
