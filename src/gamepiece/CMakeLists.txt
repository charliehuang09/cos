add_library(yolo_inference)
target_sources(yolo_inference PRIVATE
  yolo.cc
)
target_link_libraries(yolo_inference PUBLIC
  ${OpenCV_LIBS}
  CUDA::cudart
  absl::check
  absl::log
  nvinfer
  nvinfer_plugin
)
target_include_directories(yolo_inference PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

add_library(gamepiece_node)
target_sources(gamepiece_node PRIVATE
  gamepiece_node.cc
)
target_link_libraries(gamepiece_node PUBLIC
  nvjpeg_decode_node
  control_loop
  thread_pool
  nlohmann_json::nlohmann_json
  wpimath
  ${OpenCV_LIBS}
  CUDA::cudart
  absl::check
  absl::log
)
target_include_directories(gamepiece_node PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

add_library(gamepiece_control_loop)
target_sources(gamepiece_control_loop PRIVATE
  gamepiece_control_loop.cc
)
target_link_libraries(gamepiece_control_loop PUBLIC
  control_loop
  nvjpeg_decode_node
  thread_pool
  absl::check
)
target_include_directories(gamepiece_control_loop PUBLIC
  ${CMAKE_SOURCE_DIR}/src
)

add_library(gamepiece INTERFACE)
target_link_libraries(gamepiece INTERFACE
  gamepiece_control_loop
  gamepiece_node
  yolo_inference
)
