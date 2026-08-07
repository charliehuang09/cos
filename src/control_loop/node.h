#pragma once

#include <functional>

#include "control_loop/context.h"
#include "control_loop/message.h"
namespace control_loop {

// If a node declares a publication, it must setMessage when its callback is run.
// If it fails to produce a message, it should set the message to nullptr
// If a node has dependencies, it cannot assume that all the messages are there. It should call Exists to find out.
// Let X be the number of dependencies nodes a given node has. That node's callback will be called X times per control loop
class INode {
 public:
  virtual ~INode() = default;
  virtual auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> = 0;
  [[nodiscard]] virtual auto GetDependencies() const
      -> const std::vector<MessageDescriptor>& = 0;
  [[nodiscard]] virtual auto GetPublications() const
      -> const std::vector<MessageDescriptor>& = 0;
  virtual void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback) = 0;
};

}  // namespace control_loop
