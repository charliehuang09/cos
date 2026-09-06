add_library(cos.nvidia_apriltag_detector_node)
add_library(cos::nvidia_apriltag_detector_node ALIAS cos.nvidia_apriltag_detector_node)
target_sources(cos.nvidia_apriltag_detector_node
  PRIVATE
    nvidia_apriltag_detector_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      nvidia_apriltag_detector_node.h
      tag_detections.h
)

add_library(cos.cpu_apriltag_detector_node)
add_library(cos::cpu_apriltag_detector_node ALIAS cos.cpu_apriltag_detector_node)
target_sources(cos.cpu_apriltag_detector_node
  PRIVATE
    cpu_apriltag_detector_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      cpu_apriltag_detector_node.h
      tag_detections.h
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

add_library(gpu_apriltag_detector_lib)
target_sources(gpu_apriltag_detector_lib
  PRIVATE
    gpu_apriltag_detector_lib.cc
    gpu_apriltag_detector_lib.cu
)
target_compile_options(gpu_apriltag_detector_lib PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
    $<$<COMPILE_LANGUAGE:CUDA>:-diag-suppress=68,970,186>
)
target_link_libraries(gpu_apriltag_detector_lib 
  PUBLIC
    apriltag
    opencv_core
    opencv_imgcodecs
    opencv_imgproc
    timer
  PRIVATE
    CUDA::cudart
    absl::check
    absl::flat_hash_map
    absl::flat_hash_set
)
