add_library(localization_solver_common)
target_sources(localization_solver_common
  PRIVATE
    solver_common.cc
)
target_link_libraries(localization_solver_common PUBLIC
  ${OpenCV_LIBS}
  wpimath
  apriltag
  nvidia_apriltag_detector_node
)

add_library(square_solver_node)
target_sources(square_solver_node
  PRIVATE
    square_solver_node.cc
)
target_link_libraries(square_solver_node PUBLIC
  localization_solver_common
  camera_config_utils
  cv_geometry_utils
  json_utils
)

add_library(multi_tag_solver_node)
target_sources(multi_tag_solver_node
  PRIVATE
    multi_tag_solver_node.cc
)
target_link_libraries(multi_tag_solver_node PUBLIC
  localization_solver_common
  square_solver_node
  camera_config_utils
  cv_geometry_utils
  json_utils
  absl::log
)

add_library(unambiguous_solver_node)
target_sources(unambiguous_solver_node
  PRIVATE
    unambiguous_solver_node.cc
)
target_link_libraries(unambiguous_solver_node PUBLIC
  localization_solver_common
  multi_tag_solver_node
)
