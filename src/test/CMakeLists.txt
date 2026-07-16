add_executable(test_gpu_apriltag_detector)
target_sources(test_gpu_apriltag_detector
  PRIVATE
    test_gpu_apriltag_detector.cc
)

add_executable(test_thread_pool
  test_thread_pool.cc
)
target_link_libraries(test_thread_pool PRIVATE
  thread_pool
  GTest::gtest_main
)
target_link_libraries(test_gpu_apriltag_detector PRIVATE
  nvidia_apriltag_detector_node
  uvc_camera_node
  nvjpeg_decode_node
  control_loop
  thread_pool
  stop
  absl::flags_parse
  absl::log
  absl::log_initialize
)
