add_library(cos.jpeg_buffer_log_node)
add_library(cos::jpeg_buffer_log_node ALIAS cos.jpeg_buffer_log_node)
target_sources(cos.jpeg_buffer_log_node
  PRIVATE
    jpeg_buffer_log_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      jpeg_buffer_log_node.h
)
target_link_libraries(cos.jpeg_buffer_log_node PRIVATE
  absl::log
)

add_library(cos.wpilog_writer)
add_library(cos::wpilog_writer ALIAS cos.wpilog_writer)
target_sources(cos.wpilog_writer
  PRIVATE
    wpilog_writer.cc
    tag_detections_log.cc
    latency_log.cc
    estimate_log.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      wpilog_writer.h
      publication.h
      tag_detections_log.h
      latency_log.h
      estimate_log.h
)
target_include_directories(cos.wpilog_writer PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(cos.wpilog_writer PUBLIC
  absl::check
  absl::log
  wpimath
  wpiutil
  apriltag
  nlohmann_json::nlohmann_json
)
