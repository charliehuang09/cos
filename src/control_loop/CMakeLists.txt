add_library(cos.control_loop)
add_library(cos::control_loop ALIAS cos.control_loop)
target_sources(cos.control_loop
  PRIVATE
    control_loop.cc
)
target_compile_options(cos.control_loop PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)
target_link_libraries(cos.control_loop PRIVATE
  absl::log
  cos::timer
)

find_package(Threads REQUIRED)

add_library(cos.thread_pool
  thread_pool.cc
)
add_library(cos::thread_pool ALIAS cos.thread_pool)
target_link_libraries(cos.thread_pool PUBLIC
  Threads::Threads
  absl::check
)

add_library(cos.timer
  timer.cc
)
add_library(cos::timer ALIAS cos.timer)

add_library(cos.rio_clock
  rio_clock.cc
)
add_library(cos::rio_clock ALIAS cos.rio_clock)
target_link_libraries(cos.rio_clock PRIVATE
  wpilibc
  ntcore
)
