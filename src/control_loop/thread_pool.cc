#include "control_loop/thread_pool.h"

#include "absl/log/check.h"

namespace control_loop {

ThreadPool::ThreadPool(std::size_t thread_count) {
  if (thread_count == 0) {
    thread_count = 1;
  }

  workers_.reserve(thread_count);
  for (std::size_t i = 0; i < thread_count; ++i) {
    workers_.emplace_back([this]() -> void { WorkerLoop(); });
  }
}

ThreadPool::~ThreadPool() {
  Shutdown();
}

void ThreadPool::Submit(std::function<void()> task) {
  CHECK(task) << "Cannot submit an empty task";

  {
    std::lock_guard lock(mutex_);
    CHECK(accepting_tasks_) << "Cannot submit work to a stopped thread pool";
    tasks_.push(std::move(task));
  }
  work_available_.notify_one();
}

void ThreadPool::Shutdown() {
  {
    std::lock_guard lock(mutex_);
    accepting_tasks_ = false;
  }
  work_available_.notify_all();

  for (auto& worker : workers_) {
    if (worker.joinable()) {
      worker.join();
    }
  }
  workers_.clear();
}

auto ThreadPool::Size() const noexcept -> std::size_t {
  std::lock_guard lock(mutex_);
  return workers_.size();
}

void ThreadPool::WorkerLoop() {
  while (true) {
    std::function<void()> task;
    {
      std::unique_lock lock(mutex_);
      work_available_.wait(lock, [this]() -> bool {
        return !tasks_.empty() || !accepting_tasks_;
      });

      if (tasks_.empty()) {
        return;
      }

      task = std::move(tasks_.front());
      tasks_.pop();
    }
    task();
  }
}

}  // namespace control_loop
