set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests")

add_executable(test_thread_pool
  thread_pool_test.cc
)
target_link_libraries(test_thread_pool PRIVATE
  thread_pool
  GTest::gtest_main
)
add_test(NAME test_thread_pool COMMAND test_thread_pool)

add_executable(test_cv_geometry
  cv_geometry_test.cc
)
target_link_libraries(test_cv_geometry PRIVATE
  cv_geometry_utils
  GTest::gtest_main
)
add_test(NAME test_cv_geometry COMMAND test_cv_geometry)

add_executable(test_solver_common
  solver_common_test.cc
)
target_link_libraries(test_solver_common PRIVATE
  localization_solver_common
  GTest::gtest_main
)
add_test(NAME test_solver_common COMMAND test_solver_common)

add_executable(nvidia_apriltag_detector_node_test
  nvidia_apriltag_detector_node_test.cc
)
target_link_libraries(nvidia_apriltag_detector_node_test PRIVATE
  jpeg_buffer_streamer_node
  nvidia_apriltag_detector_node
  nvjpeg_fd_decode_node
  nvjpeg_decode_node
  control_loop
  thread_pool
  stop
  get_earliest_timestamp
  uvc_disk_camera_node
  absl::flags_parse
  absl::log_initialize
  uvc_camera_node
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
  square_solver_node
  multi_tag_solver_node
  nvjpeg_decode_node
  nvidia_apriltag_detector_node
  control_loop
  thread_pool
  wpilibc
  ntcore
  jpeg_buffer_streamer_node
  get_earliest_timestamp
  uvc_disk_camera_node
  camera_config
  absl::flags_parse
)

add_executable(unambiguous_solver_node_test
  unambiguous_solver_node_test.cc
)
target_link_libraries(unambiguous_solver_node_test PRIVATE
  nvjpeg_decode_node
  nvidia_apriltag_detector_node
  control_loop
  thread_pool
  wpilibc
  ntcore
  jpeg_buffer_streamer_node
  get_earliest_timestamp
  uvc_disk_camera_node
  camera_config
  absl::flags_parse
  unambiguous_solver_node
  simulation_position_sender_node
)
