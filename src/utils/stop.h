#pragma once

#include <atomic>
#include <csignal>
#include <thread>

#include "absl/log/log.h"

namespace stop {

using namespace std::literals::chrono_literals;

constexpr std::chrono::seconds kwait_interval = 1s;
constexpr std::chrono::seconds kwait_until_kill = 10s;
inline std::atomic<bool> stop(false);
inline std::atomic<bool> registered_handler(false);

inline void SignalHandler(int signal) {
  stop = true;
}

inline void RegisterHandler() {
  if (registered_handler) {
    LOG(WARNING) << "Handler has already been registred";
    return;
  }
  std::signal(SIGINT, SignalHandler);
  // std::signal(SIGILL, SignalHandler);
  // std::signal(SIGABRT, SignalHandler);
  // std::signal(SIGFPE, SignalHandler);
  // std::signal(SIGSEGV, SignalHandler);
  std::signal(SIGTERM, SignalHandler);
  std::signal(SIGHUP, SignalHandler);
  std::signal(SIGQUIT, SignalHandler);
  // std::signal(SIGTRAP, SignalHander);
  // std::signal(SIGKILL, SignalHandler);
  // std::signal(SIGPIPE, SignalHander);
  // std::signal(SIGALRM, SignalHander);

  std::thread([]() -> void {
    while (!stop) {
      std::this_thread::sleep_for(stop::kwait_interval);
    }
    std::this_thread::sleep_for(stop::kwait_until_kill);
    if (stop) {
      LOG(ERROR) << "Failed to exit cleanly";
      std::raise(SIGKILL);
    }
  }).detach();

  registered_handler = true;
}

inline void WaitUntilStop() {
  while (!stop) {
    std::this_thread::sleep_for(stop::kwait_interval);
  }
}
}  // namespace stop
