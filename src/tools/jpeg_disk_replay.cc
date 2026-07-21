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

  auto jpeg_stream_node = std::make_unique<camera::JpegDiskCamera>(
      absl::GetFlag(FLAGS_log_path), "jpeg_stream", 500);

  auto jpeg_buffer_streamer_node =
      std::make_unique<streamer::JpegBufferStreamerNode>(
          "jpeg_stream", "/stream", absl::GetFlag(FLAGS_port));

  control_loop.RegisterDependancy(jpeg_stream_node->CreateCallback());
  control_loop.RegisterCallback(jpeg_buffer_streamer_node->CreateCallback());

  control_loop.Start();

  stop::WaitUntilStop();

  control_loop.Stop();
}
