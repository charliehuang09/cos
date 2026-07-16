#include "control_loop/control_loop.h"

#include <chrono>
#include <utility>

#include "absl/log/log.h"

namespace control_loop {

ContextInternal::ContextInternal(std::chrono::steady_clock::time_point start,
                                 ControlLoop* control_loop,
                                 std::stop_token stop_token,
                                 std::atomic<bool>* destructed)
    : start(start),
      control_loop(control_loop),
      stop_token(std::move(stop_token)),
      destructed(destructed) {}

ContextInternal::~ContextInternal() {
  destructed->store(true);
  destructed->notify_all();
}

void ContextInternal::SetMessage(std::string_view path,
                                 std::unique_ptr<IMessage> message) {
  std::lock_guard lock(messages_mutex_);
  messages_.emplace(path, std::move(message));
}

ControlLoop::ControlLoop(std::chrono::milliseconds period) : period_(period) {}

void ControlLoop::Start() {
  thread_ = std::jthread([this](const std::stop_token& stop_token) -> void {
    while (!stop_token.stop_requested()) {
      VLOG(1) << "Control loop";
      std::stop_source stop_source;
      std::atomic destructed = false;

      Context context(new ContextInternal(std::chrono::steady_clock::now(),
                                          this, stop_source.get_token(),
                                          &destructed));

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
