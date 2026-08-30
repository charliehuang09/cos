add_library(cos.nvidia_apriltag_detector_node)
add_library(cos::nvidia_apriltag_detector_node ALIAS cos.nvidia_apriltag_detector_node)
target_sources(cos.nvidia_apriltag_detector_node
  PRIVATE
    nvidia_apriltag_detector_node.cc
)

add_library(cos.cpu_apriltag_detector_node)
add_library(cos::cpu_apriltag_detector_node ALIAS cos.cpu_apriltag_detector_node)
target_sources(cos.cpu_apriltag_detector_node PRIVATE
  cpu_apriltag_detector_node.cc
)
target_link_libraries(cos.cpu_apriltag_detector_node PUBLIC
  apriltag
  ${OpenCV_LIBS}
  nlohmann_json::nlohmann_json
  absl::log
)
target_link_libraries(cos.nvidia_apriltag_detector_node
  PUBLIC
    ${OpenCV_LIBS}
    vpi
    cos::nvjpeg_fd_decode_node
    cos::nvjpeg_decode_node
  PRIVATE
    cos::cuda_utils
    nlohmann_json::nlohmann_json

)
