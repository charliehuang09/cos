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

ABSL_FLAG(uint, skip_frame_frequency, 0,  // NOLINT
          "Skip each replayed frame with probability 1/N; zero disables "
          "skipping");  // NOLINT

ABSL_FLAG(uint, empty_frame_frequency, 0,  // NOLINT
          "Publish an empty frame every Nth replayed frame; zero disables "
          "it");  // NOLINT

auto main(int argc, char* argv[]) -> int {
  absl::ParseCommandLine(argc, argv);

  CHECK(absl::GetFlag(FLAGS_log_path) != "");
  stop::RegisterHandler();

  control_loop::ControlLoop control_loop(15ms);

  {

    auto jpeg_disk_camera_node = std::make_shared<camera::JpegDiskCamera>(
        absl::GetFlag(FLAGS_log_path), "jpeg_stream", true, false,
        absl::GetFlag(FLAGS_skip_frame_frequency),
        absl::GetFlag(FLAGS_empty_frame_frequency));

    auto jpeg_buffer_streamer_node =
        std::make_shared<streamer::JpegBufferStreamerNode>(
            "jpeg_stream", "/stream", absl::GetFlag(FLAGS_port));

    control_loop.RegisterDependancyNode(jpeg_disk_camera_node);
    control_loop.RegisterNode(jpeg_buffer_streamer_node);
  }

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
}
