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

add_executable(nvidia_apriltag_detect)
target_sources(nvidia_apriltag_detect
  PRIVATE
    nvidia_apriltag_detect.cc
)
target_link_libraries(nvidia_apriltag_detect PRIVATE
  nvidia_apriltag_detector_node
)

