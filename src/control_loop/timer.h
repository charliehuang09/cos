#pragma once
#include <chrono>
namespace control_loop {
// Utility class used to get the amount of time it takes for a block of code to be run.
// Timer starts when the object is construcuted and stops when Stop() is called or when
// the class gets destructed (useally when the object gets out of scope)
class Timer {
 public:
  Timer();
  auto Stop() -> std::chrono::duration<double>;

 private:
  std::chrono::steady_clock::time_point start_;
};

}  // namespace control_loop
