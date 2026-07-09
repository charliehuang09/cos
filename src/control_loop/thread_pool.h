#pragma once

#include <cstddef>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace control_loop {

// A fixed-size pool of worker threads. Queued work is completed before shutdown.
class ThreadPool {
 public:
  explicit ThreadPool(
      std::size_t thread_count = std::thread::hardware_concurrency());
  ~ThreadPool();

  ThreadPool(const ThreadPool&) = delete;
  auto operator=(const ThreadPool&) -> ThreadPool& = delete;
  ThreadPool(ThreadPool&&) = delete;
  auto operator=(ThreadPool&&) -> ThreadPool& = delete;

  void Submit(std::function<void()> task);

  // Stops accepting work, finishes queued tasks, and joins all workers.
  // Calling Shutdown more than once is safe.
  void Shutdown();

  [[nodiscard]] auto Size() const noexcept -> std::size_t;

 private:
  void WorkerLoop();

  mutable std::mutex mutex_;
  std::condition_variable work_available_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::thread> workers_;
  bool accepting_tasks_ = true;
};

}  // namespace control_loop
