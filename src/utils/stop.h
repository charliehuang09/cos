#pragma once

#include <atomic>
#include <chrono>
#include <csignal>
#include <thread>

#include "absl/log/log.h"

namespace stop {

using namespace std::literals::chrono_literals;

constexpr std::chrono::seconds kwait_interval = 1s;
constexpr std::chrono::seconds kwait_until_kill = 10s;
// A signal handler may only modify volatile sig_atomic_t state.  Normal
// threads communicate through stop, which must remain atomic.
inline volatile std::sig_atomic_t signal_stop = 0;
inline std::atomic<bool> stop(false);
inline std::atomic<bool> registered_handler(false);

inline void SignalHandler(int /*signal*/) {
  signal_stop = 1;
}

inline void RequestStop() {
  stop.store(true, std::memory_order_relaxed);
}

inline auto StopRequested() -> bool {
  if (signal_stop != 0) {
    RequestStop();
  }
  return stop.load(std::memory_order_relaxed);
}

inline void RegisterHandler() {
  bool expected = false;
  if (!registered_handler.compare_exchange_strong(expected, true)) {
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
    while (!StopRequested()) {
      std::this_thread::sleep_for(stop::kwait_interval);
    }
    std::this_thread::sleep_for(stop::kwait_until_kill);
    if (StopRequested()) {
      LOG(ERROR) << "Failed to exit cleanly";
      std::raise(SIGKILL);
    }
  }).detach();

  registered_handler = true;
}

inline void WaitUntilStop() {
  while (!StopRequested()) {
    std::this_thread::sleep_for(stop::kwait_interval);
  }
}
}  // namespace stop
