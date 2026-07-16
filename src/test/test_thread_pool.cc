#include "control_loop/thread_pool.h"

#include <atomic>
#include <future>

#include "gtest/gtest.h"

namespace control_loop {
namespace {

TEST(ThreadPoolTest, ExecutesSubmittedTasks) {
  ThreadPool pool(2);
  std::promise<int> result;

  pool.Submit([&result]() -> void { result.set_value(42); });

  EXPECT_EQ(result.get_future().get(), 42);
}

TEST(ThreadPoolTest, DrainsQueuedTasksOnShutdown) {
  ThreadPool pool(2);
  std::atomic<int> completed = 0;
  for (int i = 0; i < 100; ++i) {
    pool.Submit([&completed]() -> void { ++completed; });
  }

  pool.Shutdown();

  EXPECT_EQ(completed, 100);
  EXPECT_DEATH(pool.Submit([] {}), "stopped thread pool");  // NOLINT
}

TEST(ThreadPoolTest, RejectsEmptyTasks) {
  ThreadPool pool(1);

  EXPECT_DEATH(pool.Submit({}), "empty task");
}

TEST(ThreadPoolTest, UsesOneWorkerWhenCountIsZero) {
  ThreadPool pool(0);
  std::promise<void> completed;

  EXPECT_EQ(pool.Size(), 1U);
  pool.Submit([&completed]() -> void { completed.set_value(); });
  completed.get_future().wait();
}

}  // namespace
}  // namespace control_loop
