add_executable(uvc_logger)
target_sources(uvc_logger
  PRIVATE
    uvc_logger.cc
)
target_link_libraries(uvc_logger PRIVATE
  jpeg_buffer_streamer_node
  uvc_camera_node
  stop
  absl::flags_parse
  nvjpeg_decode_node
  JPEG::JPEG
  ${OpenCV_LIBS}
)
