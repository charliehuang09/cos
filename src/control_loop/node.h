#pragma once

#include <functional>

#include "control_loop/context.h"
#include "control_loop/message.h"
namespace control_loop {

class INode {
 public:
  virtual ~INode() = default;
  virtual auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> = 0;
  [[nodiscard]] virtual auto GetDependencies() const
      -> const std::vector<MessageDescriptor>& = 0;
  [[nodiscard]] virtual auto GetPublications() const
      -> const std::vector<MessageDescriptor>& = 0;
};

}  // namespace control_loop
