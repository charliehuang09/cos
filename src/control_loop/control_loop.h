#pragma once

#include <chrono>
#include <functional>
#include <thread>
#include <vector>

#include "control_loop/context.h"
#include "control_loop/node.h"

namespace control_loop {

class ControlLoop {
 public:
  ControlLoop(std::chrono::milliseconds frequency);
  void RegisterCallback(std::function<void(const Context&)> callback);
  void RegisterDependancy(std::function<void(const Context&)>);
  void RegisterNode(const std::shared_ptr<INode>& node);
  void RegisterDependancyNode(const std::shared_ptr<INode>& node);
  void Start();
  void Stop();

 private:
  void ValidateNodeGraph();
  void RegisterNodeCallbacks();

 private:
  std::jthread thread_;
  std::chrono::milliseconds period_;
  std::vector<std::function<void(Context)>> callbacks_;
  std::vector<std::function<void(Context)>> dependencies_;
  std::vector<std::shared_ptr<INode>> nodes_;
  std::vector<std::shared_ptr<INode>> dependancy_nodes_;
};

}  // namespace control_loop
