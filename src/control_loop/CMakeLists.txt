add_library(control_loop)
target_sources(control_loop
  PRIVATE
    control_loop.cc
)
target_compile_options(control_loop PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)
target_link_libraries(control_loop PRIVATE
  absl::log
)
