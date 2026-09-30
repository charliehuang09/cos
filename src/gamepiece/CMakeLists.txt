add_library(cos.lane_density)
add_library(cos::lane_density ALIAS cos.lane_density)
target_sources(cos.lane_density
  PRIVATE
    lane_density.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      lane_density.h
)
target_link_libraries(cos.lane_density PUBLIC
  cos::camera_config
)
target_link_libraries(cos.lane_density PRIVATE
  cos::image_utils
  cos::cv_geometry_utils
  opencv_calib3d
  absl::flags
  absl::log
)
