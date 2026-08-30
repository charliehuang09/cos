add_library(cos.jpeg_buffer_log_node)
add_library(cos::jpeg_buffer_log_node ALIAS cos.jpeg_buffer_log_node)
target_sources(cos.jpeg_buffer_log_node
  PRIVATE
    jpeg_buffer_log_node.cc
)
target_link_libraries(cos.jpeg_buffer_log_node PRIVATE
  absl::log
)
