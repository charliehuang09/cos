add_library(gpu_apriltag_detector_node)
target_sources(gpu_apriltag_detector_node
  PRIVATE
    gpu_apriltag_detector_node.cc
)
target_link_libraries(gpu_apriltag_detector_node PUBLIC
  nlohmann_json::nlohmann_json
)

add_library(nvidia_apriltag_detector_node)
target_sources(nvidia_apriltag_detector_node
  PRIVATE
    nvidia_apriltag_detector_node.cc
)
target_link_libraries(nvidia_apriltag_detector_node 
  PUBLIC
    ${OpenCV_LIBS}
    vpi
    nvjpeg_decode_node
  PRIVATE
    nlohmann_json::nlohmann_json

)
