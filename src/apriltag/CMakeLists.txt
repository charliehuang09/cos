add_library(nvidia_apriltag_detector_node)
target_sources(nvidia_apriltag_detector_node
  PRIVATE
    nvidia_apriltag_detector_node.cc
)

add_library(cpu_apriltag_detector_node)
target_sources(cpu_apriltag_detector_node PRIVATE
  cpu_apriltag_detector_node.cc
)
target_link_libraries(cpu_apriltag_detector_node PUBLIC
  apriltag
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  absl::log
)
target_link_libraries(nvidia_apriltag_detector_node
  PUBLIC
    ${OpenCV_LIBS}
    vpi
    nvjpeg_fd_decode_node
    nvjpeg_decode_node
  PRIVATE
    cuda_utils
    nlohmann_json::nlohmann_json

)
