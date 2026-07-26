add_executable(test_thread_pool
  thread_pool_test.cc
)
target_link_libraries(test_thread_pool PRIVATE
  thread_pool
  GTest::gtest_main
)
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
  jpeg_disk_camera
  absl::log_initialize
)

add_executable(full_pipeline_localization_test
  full_pipeline_localization_test.cc
)
target_link_libraries(full_pipeline_localization_test PRIVATE
  jpeg_disk_camera
  cpu_decode_node
  nvjpeg_decode_node
  cpu_apriltag_detector_node
  nvidia_apriltag_detector_node
  joint_solver_node
  control_loop
  thread_pool
  wpiutil
  absl::flags_parse
  absl::log
  absl::log_initialize
)
add_executable(timestamp_test
  timestamp_test.cc
)
target_link_libraries(timestamp_test PRIVATE
  wpilibc
  ntcore
)

add_executable(joint_solver_node_test
  joint_solver_node_test.cc
)
target_compile_definitions(joint_solver_node_test PRIVATE
  TEST_CONSTANTS_DIRECTORY="${CMAKE_CURRENT_SOURCE_DIR}/test_constants/main_bot"
)
target_link_libraries(joint_solver_node_test PRIVATE
  joint_solver_node
  camera_config_utils
  control_loop
  cv_geometry_utils
  json_utils
)
