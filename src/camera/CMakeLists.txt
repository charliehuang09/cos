add_library(uvc_camera_node)
target_sources(uvc_camera_node
  PRIVATE
    uvc_camera_node.cc
)
target_link_libraries(uvc_camera_node PRIVATE
  uvc
  absl::log
  nlohmann_json::nlohmann_json
)

add_library(nvjpeg_decode_node)
target_sources(nvjpeg_decode_node
  PRIVATE
    nvjpeg_decode_node.cc
)
target_compile_options(nvjpeg_decode_node PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)
target_link_libraries(nvjpeg_decode_node PUBLIC
  CUDA::cudart
  CUDA::nvjpeg
  cuda_utils
  absl::log
)
