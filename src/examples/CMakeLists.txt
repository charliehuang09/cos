set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/examples")

add_executable(mjpeg_streamer)
target_sources(mjpeg_streamer
  PRIVATE
    mjpeg_streamer.cc
)
target_link_libraries(mjpeg_streamer PRIVATE
  nadjieb_mjpeg_streamer::nadjieb_mjpeg_streamer
  ${OpenCV_LIBS}
)
target_include_directories(mjpeg_streamer PRIVATE
  ${OpenCV_INCLUDE_DIRS}
)

add_executable(control_loop_example)
target_sources(control_loop_example
  PRIVATE
    control_loop_example.cc
)
target_link_libraries(control_loop_example PRIVATE
  control_loop
  absl::log_initialize
  absl::flags_parse
  $<LINK_LIBRARY:WHOLE_ARCHIVE,absl::log_flags>
)
target_include_directories(control_loop_example PRIVATE
)

add_executable(nvidia_apriltag_detect)
target_sources(nvidia_apriltag_detect
  PRIVATE
    nvidia_apriltag_detect.cc
)
target_link_libraries(nvidia_apriltag_detect PRIVATE
  nvidia_apriltag_detector_node
  absl::flags_parse
  absl::log
  absl::log_initialize
  $<LINK_LIBRARY:WHOLE_ARCHIVE,absl::log_flags>
)

add_executable(visualization_backend)
target_sources(visualization_backend PRIVATE
  visualization_backend.cc
)
target_link_libraries(visualization_backend PRIVATE
  ixwebsocket
)

add_executable(gpu_apriltag)
target_sources(gpu_apriltag PRIVATE
  gpu_apriltag.cc
)
target_link_libraries(gpu_apriltag PRIVATE
  timer
  gpu_apriltag_detector_lib
  absl::log
  absl::check
  absl::flags_parse
  absl::log_initialize
  ${OpenCV_LIBS}
)
