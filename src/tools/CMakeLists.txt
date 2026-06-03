add_executable(uvc_streamer)
target_sources(uvc_streamer
  PRIVATE
    uvc_streamer.cc
)
target_link_libraries(uvc_streamer PRIVATE
  jpeg_buffer_streamer_node
  uvc_camera_node
  stop
  absl::flags_parse
)
