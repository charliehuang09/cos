#include "control_loop/control_loop.h"

#include <chrono>

#include "absl/log/log.h"

namespace control_loop {

ContextInternal::~ContextInternal() {
  destructed->store(true);
  destructed->notify_all();
}

ControlLoop::ControlLoop(std::chrono::milliseconds period) : period_(period) {}

void ControlLoop::Start() {
  thread_ = std::jthread([this](const std::stop_token& stop_token) -> void {
    while (!stop_token.stop_requested()) {
      VLOG(1) << "Control loop";
      std::stop_source stop_source;
      std::atomic destructed = false;

      Context context(new ContextInternal{
          .start = std::chrono::steady_clock::now(),
          .control_loop = this,
          .stop_token = stop_source.get_token(),
          .destructed = &destructed,
      });

      for (const auto& dependancy : dependencies_) {
        dependancy(context);
      }

      for (const auto& callback : callbacks_) {
        callback(context);
      }

      context.reset();

      std::this_thread::sleep_for(period_);

      if (!destructed) {
        LOG(WARNING) << "Command loop overun";
        stop_source.request_stop();
        destructed.wait(false);
      }
    }
  });
}

void ControlLoop::Stop() {
  thread_.request_stop();
  if (thread_.joinable()) {
    thread_.join();
  }
}

void ControlLoop::RegisterCallback(
    std::function<void(const Context&)> callback) {
  callbacks_.emplace_back(callback);
}

void ControlLoop::RegisterDependancy(
    std::function<void(const Context&)> dependancy) {
  dependencies_.emplace_back(dependancy);
}

}  // namespace control_loop
