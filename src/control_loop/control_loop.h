#pragma once

#include <chrono>
#include <functional>
#include <thread>
#include <vector>

#include "control_loop/context.h"

namespace control_loop {

class ControlLoop {
 public:
  ControlLoop(std::chrono::milliseconds frequency);
  void RegisterCallback(std::function<void(const Context&)> callback);
  void RegisterDependancy(std::function<void(const Context&)>);
  void Start();
  void Stop();

 private:
  std::jthread thread_;
  std::chrono::milliseconds period_;
  std::vector<std::function<void(Context)>> callbacks_;
  std::vector<std::function<void(Context)>> dependencies_;
};

}  // namespace control_loop
