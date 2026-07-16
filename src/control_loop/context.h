#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <unordered_map>

#include "control_loop/message.h"

namespace control_loop {

class ControlLoop;

struct ContextInternal {
  ContextInternal(std::chrono::steady_clock::time_point start,
                  ControlLoop* control_loop, std::stop_token stop_token,
                  std::atomic<bool>* destructed);
  ~ContextInternal();

  template <typename T>
  auto GetMessage(std::string_view path) const -> T* {
    std::lock_guard lock(messages_mutex_);
    const auto message_it = messages_.find(std::string(path));
    if (message_it == messages_.end()) {
      return nullptr;
    }
    return dynamic_cast<T*>(message_it->second.get());
  }

  void SetMessage(std::string_view path, std::unique_ptr<IMessage> message);

  std::optional<std::chrono::steady_clock::time_point> start;
  ControlLoop* control_loop;
  std::stop_token stop_token;
  std::atomic<bool>* destructed;

 private:
  mutable std::mutex messages_mutex_;
  std::unordered_map<std::string, std::unique_ptr<IMessage>> messages_;
};

using Context = std::shared_ptr<ContextInternal>;

}  // namespace control_loop
