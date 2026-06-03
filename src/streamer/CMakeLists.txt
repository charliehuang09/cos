add_library(jpeg_buffer_streamer_node)
target_sources(jpeg_buffer_streamer_node
  PRIVATE
    jpeg_buffer_streamer_node.cc
)
target_link_libraries(jpeg_buffer_streamer_node PUBLIC 
  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
  absl::log_initialize
)
