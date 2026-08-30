set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

add_executable(test_thread_pool
  thread_pool_test.cc
)
target_link_libraries(test_thread_pool PRIVATE
  cos::thread_pool
  GTest::gtest_main
)

add_executable(test_cv_geometry
  cv_geometry_test.cc
)
target_link_libraries(test_cv_geometry PRIVATE
  cos::cv_geometry_utils
  GTest::gtest_main
)

add_executable(test_solver_common
  solver_common_test.cc
)
target_link_libraries(test_solver_common PRIVATE
  cos::localization_solver_common
  GTest::gtest_main
)

add_executable(nvidia_apriltag_detector_node_test
  nvidia_apriltag_detector_node_test.cc
)
target_link_libraries(nvidia_apriltag_detector_node_test PRIVATE
  cos::jpeg_buffer_streamer_node
  cos::nvidia_apriltag_detector_node
  cos::nvjpeg_fd_decode_node
  cos::nvjpeg_decode_node
  cos::control_loop
  cos::thread_pool
  cos::stop
  cos::get_earliest_timestamp
  cos::uvc_disk_camera_node
  absl::flags_parse
  absl::log_initialize
  cos::uvc_camera_node
)

add_executable(timestamp_test
  timestamp_test.cc
)
target_link_libraries(timestamp_test PRIVATE
  wpilibc
  ntcore
)

add_executable(square_solve_node_test
  square_solver_node_test.cc
)
target_link_libraries(square_solve_node_test PRIVATE
  cos::square_solver_node
  cos::multi_tag_solver_node
  cos::nvjpeg_decode_node
  cos::nvidia_apriltag_detector_node
  cos::control_loop
  cos::thread_pool
  wpilibc
  ntcore
  cos::jpeg_buffer_streamer_node
  cos::get_earliest_timestamp
  cos::uvc_disk_camera_node
  cos::camera_config
  absl::flags_parse
)

add_executable(unambiguous_solver_node_test
  unambiguous_solver_node_test.cc
)
target_link_libraries(unambiguous_solver_node_test PRIVATE
  cos::nvjpeg_decode_node
  cos::nvidia_apriltag_detector_node
  cos::control_loop
  cos::thread_pool
  wpilibc
  ntcore
  cos::jpeg_buffer_streamer_node
  cos::get_earliest_timestamp
  cos::uvc_disk_camera_node
  cos::camera_config
  absl::flags_parse
  cos::unambiguous_solver_node
  cos::simulation_position_sender_node
)
