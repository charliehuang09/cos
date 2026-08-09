add_library(jpeg_buffer_log_node)
target_sources(jpeg_buffer_log_node
  PRIVATE
    jpeg_buffer_log_node.cc
)
target_link_libraries(jpeg_buffer_log_node PRIVATE
  absl::log
)

add_library(wpilog_writer)
target_sources(wpilog_writer PRIVATE
  wpilog_writer.cc
  tag_detections_log.cc
)
target_link_libraries(wpilog_writer PUBLIC
  absl::log
  wpilibc
)
