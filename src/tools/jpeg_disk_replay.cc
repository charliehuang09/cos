#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"

#include "camera/jpeg_disk_camera.h"
#include "control_loop/control_loop.h"
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

  control_loop::ControlLoop control_loop(15ms);

  {

    auto jpeg_disk_camera_node = std::make_shared<camera::JpegDiskCamera>(
        absl::GetFlag(FLAGS_log_path), "jpeg_stream");

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>(
            "jpeg_stream", "/stream", absl::GetFlag(FLAGS_port));

    control_loop.RegisterDependancyNode(jpeg_disk_camera_node);
    jpeg_disk_camera_node->RegisterCallback(
        [jpeg_buffer_streamer_node](
            const control_loop::Context& context) -> void {
          jpeg_buffer_streamer_node->RegisterCallback(context);
        });
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
}
