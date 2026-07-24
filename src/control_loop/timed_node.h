#pragma once

#include "control_loop/message.h"

#include <chrono>
#include <string_view>
namespace control_loop {

class ITimedNode {
 public:
  virtual void EnableTiming(
      std::string_view
          latency_channel) = 0;  // Channel type should be std::chrono::duration<double> (seconds)
};

class LatencyMessage final : public control_loop::IMessage {
 public:
  LatencyMessage(std::chrono::duration<double> latency_) : latency(latency_) {}
  auto GetType() -> const std::type_info& override {
    return typeid(LatencyMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }

  std::chrono::duration<double> latency;
};

}  // namespace control_loop
