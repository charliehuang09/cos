#include "control_loop/rio_clock.h"
#include "absl/log/log.h"

#include <cstdint>
#include <limits>

#include "networktables/NetworkTableInstance.h"
#include "wpi/timestamp.h"

namespace control_loop {
std::atomic<bool> RioClock::simulation_{false};
std::chrono::steady_clock::time_point RioClock::start_time_{
    std::chrono::steady_clock::now()};

RioClock::RioClock() = default;

void RioClock::EnableSimulation() {
  Restart();
  simulation_ = true;
}

void RioClock::DisableSimulation() {
  simulation_ = false;
}

auto RioClock::GetInstance() -> RioClock& {
  static RioClock instance;
  return instance;
}

auto RioClock::GetTime() -> double {
  if (simulation_) {
    const auto time = std::chrono::steady_clock::now() - start_time_;
    return std::chrono::duration<double>(time).count();
  } else {
    const auto offset =
        nt::NetworkTableInstance::GetDefault().GetServerTimeOffset();
    if (!offset) {
      return std::numeric_limits<double>::quiet_NaN();
    }
    // Both values are microseconds. The offset is signed and can be negative
    // when translating the Orin's Unix epoch to the roboRIO's FPGA epoch.
    const auto server_time = static_cast<int64_t>(wpi::Now()) + *offset;
    const double tmp = static_cast<double>(server_time) / 1'000'000.0;
    return tmp;
  }
}

void RioClock::Restart() {
  start_time_ = std::chrono::steady_clock::now();
}

};  // namespace control_loop
