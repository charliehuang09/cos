add_library(jpeg_buffer_log_node)
target_sources(jpeg_buffer_log_node
  PRIVATE
    jpeg_buffer_log_node.cc
)
target_link_libraries(jpeg_buffer_log_node PRIVATE
  absl::log
)

