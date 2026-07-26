#pragma once

#include <chrono>
#include <functional>
#include <queue>
#include <thread>
#include <vector>

#include "control_loop/context.h"
#include "control_loop/node.h"

namespace control_loop {

class ControlLoop {
 public:
  ControlLoop() = default;
  ControlLoop(std::chrono::milliseconds frequency);
  void RegisterCallback(const std::function<void(const Context&)>& callback);
  void RegisterDependancy(const std::function<void(const Context&)>&);
  void RegisterNode(const std::shared_ptr<INode>& node);
  void RegisterDependancyNode(const std::shared_ptr<INode>& node);
  void EnableLatencyLog();
  void Start();
  void Stop();
  [[nodiscard]] auto GetLoopsPerSecond() const -> double;

 private:
  void ValidateNodeGraph();
  void RegisterNodeCallbacks();

 private:
  std::jthread thread_;
  std::optional<std::chrono::milliseconds> period_;
  std::vector<std::function<void(Context)>> callbacks_;
  std::vector<std::function<void(Context)>> dependencies_;
  std::vector<std::shared_ptr<INode>> nodes_;
  std::vector<std::shared_ptr<INode>> dependancy_nodes_;
  bool log_latency_ = false;
  std::queue<std::chrono::steady_clock::time_point> timestamp_queue_;
  std::atomic<double> loops_per_second_ = -1;

 private:
  static const size_t kTimestampQueueMaxSize = 100;
  constexpr static const double kMinLoopSeconds = 0.001;
};

}  // namespace control_loop
