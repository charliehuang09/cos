#pragma once

#include <functional>

#include "control_loop/context.h"
namespace control_loop {

class INode {
 public:
  virtual ~INode() = default;
  virtual auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> = 0;
};

}  // namespace control_loop
