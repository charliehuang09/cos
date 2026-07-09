find_package(Threads REQUIRED)

add_library(thread_pool
  thread_pool.cc
)
target_link_libraries(thread_pool PUBLIC
  Threads::Threads
  absl::check
)
