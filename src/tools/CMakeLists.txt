set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tools")

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
  cuda_utils
  wpilibc
  jpeg_buffer_log_node
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

add_executable(intrinsics_calibrate)
target_sources(intrinsics_calibrate
  PRIVATE
    intrinsics_calibrate.cc
)
target_link_libraries(intrinsics_calibrate PRIVATE
  uvc_camera_node
  control_loop
  stop
  absl::check
  absl::flags_parse
  absl::log_initialize
  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
  nlohmann_json::nlohmann_json
  ${OpenCV_LIBS}
)
target_include_directories(intrinsics_calibrate PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)
