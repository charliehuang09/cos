add_library(stop)
target_sources(stop
  PRIVATE
    stop.cc
)
target_link_libraries(stop PUBLIC 
  absl::log
)
