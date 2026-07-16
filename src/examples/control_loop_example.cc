#include "absl/flags/parse.h"
#include "control_loop/control_loop.h"

#include <thread>

#include "absl/log/initialize.h"
#include "absl/log/log.h"

using namespace std::chrono_literals;

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();

  VLOG(1) << "asdfasdfasdf";

  control_loop::ControlLoop control_loop(20ms);

  control_loop.Start();

  std::this_thread::sleep_for(1000ms);
}
