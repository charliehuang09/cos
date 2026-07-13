#pragma once

#include <chrono>
#include <functional>
#include <memory>
#include <optional>
#include <thread>
#include <vector>

#include "control_loop/message.h"

namespace control_loop {

class ControlLoop;

struct ContextInternal {
  std::optional<std::chrono::steady_clock::time_point> start;
  ControlLoop* control_loop;
  std::stop_token stop_token;
  std::atomic<bool>* destructed;
  std::unordered_map<std::string, std::unique_ptr<IMessage>> messages;
  ~ContextInternal();
};

using Context = std::shared_ptr<ContextInternal>;

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
