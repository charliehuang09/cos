#include "control_loop/timer.h"
#include <chrono>
namespace control_loop {
Timer::Timer() : start_(std::chrono::steady_clock::now()) {}

auto Timer::Stop() -> std::chrono::duration<double> {
  auto end = std::chrono::steady_clock::now();
  return end - start_;
}

auto Timer::GetStart() -> std::chrono::steady_clock::time_point {
  return start_;
}

}  // namespace control_loop
