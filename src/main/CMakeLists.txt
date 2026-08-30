set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/main")

add_executable(dev_orin_main)
target_sources(dev_orin_main
  PRIVATE
    dev_orin_main.cc
)
target_link_libraries(dev_orin_main PRIVATE
  cos::jpeg_buffer_streamer_node
  cos::uvc_camera_node
  cos::stop
  absl::flags_parse
  cos::nvjpeg_decode_node
  JPEG::JPEG
  cos::control_loop
  cos::thread_pool
  cos::cuda_utils
  wpilibc
  cos::nvidia_apriltag_detector_node
  cos::unambiguous_solver_node
  cos::position_estimate_sender_node
  cos::jpeg_buffer_log_node
  cos::simulation_position_sender_node
  ${OpenCV_LIBS}
)

add_executable(second_bot_main)
target_sources(second_bot_main
  PRIVATE
    second_bot_main.cc
)
target_link_libraries(second_bot_main PRIVATE
  cos::jpeg_buffer_streamer_node
  cos::get_earliest_timestamp
  cos::uvc_disk_camera_node
  cos::uvc_camera_node
  cos::stop
  absl::flags_parse
  cos::nvjpeg_decode_node
  JPEG::JPEG
  cos::control_loop
  cos::thread_pool
  cos::cuda_utils
  wpilibc
  cos::nvidia_apriltag_detector_node
  cos::unambiguous_solver_node
  cos::position_estimate_sender_node
  cos::jpeg_buffer_log_node
  cos::simulation_position_sender_node
  ${OpenCV_LIBS}
)
