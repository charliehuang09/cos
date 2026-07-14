#pragma once

#include "control_loop/control_loop.h"
namespace control_loop {

class INode {
 public:
  virtual ~INode() = default;
  virtual auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> = 0;
};

}  // namespace control_loop
