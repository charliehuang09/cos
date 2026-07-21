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
  control_loop
  thread_pool
  wpilibc
  ${OpenCV_LIBS}
)

add_executable(compress_jpeg)
target_sources(compress_jpeg
  PRIVATE
    compress_jpeg.cc
)
target_link_libraries(compress_jpeg PRIVATE
  absl::flags_parse
  absl::check
  ${OpenCV_LIBS}
)

add_executable(jpeg_disk_replay)
target_sources(jpeg_disk_replay
  PRIVATE
    jpeg_disk_replay.cc
)
target_link_libraries(jpeg_disk_replay PRIVATE
  jpeg_disk_camera
  jpeg_buffer_streamer_node
  stop
  absl::flags_parse
  control_loop
  thread_pool
)
