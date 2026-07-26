find_library(NVMM_JPEG_TEST_LIBRARY NAMES nvmm_jpeg REQUIRED)

if(CMAKE_CROSSCOMPILING)
  set(COS_JPEG_TEST_IMAGE "/cos-logs/log181/right/6.675735.jpg")
  set(COS_POSE_TEST_CONSTANTS "/root/test_constants/main_bot")
else()
  set(COS_JPEG_TEST_IMAGE "${CMAKE_SOURCE_DIR}/bos-logs/log181/right/6.675735.jpg")
  set(COS_POSE_TEST_CONSTANTS "${CMAKE_CURRENT_SOURCE_DIR}/test_constants/main_bot")
endif()

add_executable(test_thread_pool
  thread_pool_test.cc
)
target_link_libraries(test_thread_pool PRIVATE
  thread_pool
  GTest::gtest_main
)
add_test(NAME test_thread_pool COMMAND test_thread_pool)
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

add_executable(jpeg_decoder_node_test
  jpeg_decoder_node_test.cc
)
target_compile_definitions(jpeg_decoder_node_test PRIVATE
  JPEG_TEST_IMAGE="${COS_JPEG_TEST_IMAGE}"
)
target_link_libraries(jpeg_decoder_node_test PRIVATE
  jetson_mmjpeg
  ${NVMM_JPEG_TEST_LIBRARY}
  cpu_decode_node
  nvjpeg_decode_node
  nvjpeg_fd_decode_node
  control_loop
  thread_pool
  GTest::gtest_main
)
add_test(NAME jpeg_decoder_node_test COMMAND jpeg_decoder_node_test)

add_executable(pose_solver_node_test
  pose_solver_node_test.cc
)
target_compile_definitions(pose_solver_node_test PRIVATE
  POSE_TEST_CONSTANTS="${COS_POSE_TEST_CONSTANTS}"
)
target_link_libraries(pose_solver_node_test PRIVATE
  square_solver_node
  multi_tag_solver_node
  unambiguous_solver_node
  joint_solver_node
  control_loop
  GTest::gtest_main
)
add_test(NAME pose_solver_node_test COMMAND pose_solver_node_test)

add_executable(node_contract_test
  node_contract_test.cc
)
target_link_libraries(node_contract_test PRIVATE
  jpeg_disk_camera
  jpeg_buffer_streamer_node
  jpeg_buffer_log_node
  control_loop
  thread_pool
  GTest::gtest_main
)
add_test(NAME node_contract_test COMMAND node_contract_test)

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

add_executable(gamepiece_control_loop_test
  gamepiece_control_loop_test.cc
)
target_link_libraries(gamepiece_control_loop_test PRIVATE
  gamepiece_control_loop
  GTest::gtest_main
)

add_executable(full_gamepiece_pipeline_test
  full_gamepiece_pipeline_test.cc
)
target_link_libraries(full_gamepiece_pipeline_test PRIVATE
  jpeg_disk_camera
  nvjpeg_decode_node
  gamepiece_node
  yolo_inference
  control_loop
  thread_pool
  json_utils
  absl::flags_parse
  absl::log
  absl::log_initialize
)
