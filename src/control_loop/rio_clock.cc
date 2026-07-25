#include "control_loop/rio_clock.h"

#include "frc/Timer.h"

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
    return frc::Timer::GetFPGATimestamp().to<double>();
  }
}

void RioClock::Restart() {
  start_time_ = std::chrono::steady_clock::now();
}

};  // namespace control_loop
