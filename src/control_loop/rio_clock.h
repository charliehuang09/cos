#pragma once

#include <atomic>
#include <chrono>
namespace control_loop {
class RioClock {
 public:
  static auto GetInstance() -> RioClock&;
  static auto GetTime() -> double;
  static void EnableSimulation();
  static void DisableSimulation();
  static void Restart();

  RioClock(const RioClock&) = delete;
  auto operator=(const RioClock&) -> RioClock& = delete;
  RioClock(RioClock&&) = delete;
  auto operator=(RioClock&&) -> RioClock& = delete;

 private:
  RioClock();

 private:
  static std::atomic<bool> simulation_;
  static std::chrono::steady_clock::time_point start_time_;
};
}  // namespace control_loop
