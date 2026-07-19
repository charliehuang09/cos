add_library(stop)
target_sources(stop
  PRIVATE
    stop.cc
)
target_link_libraries(stop PUBLIC
  absl::log
)

add_library(cuda_utils)
target_sources(cuda_utils
  PRIVATE
    cuda.cc
)
target_link_libraries(cuda_utils PUBLIC
  CUDA::cudart
  absl::check
)

add_library(json_utils)
target_sources(json_utils
  PRIVATE
    json.cc
)
target_link_libraries(json_utils PUBLIC
  nlohmann_json::nlohmann_json
  absl::check
)

add_library(camera_config_utils)
target_sources(camera_config_utils
  PRIVATE
    camera_config.cc
)
target_link_libraries(camera_config_utils PUBLIC
  ${OpenCV_LIBS}
  wpimath
  nlohmann_json::nlohmann_json
)

add_library(cv_geometry_utils)
target_sources(cv_geometry_utils
  PRIVATE
    cv_geometry.cc
)
target_link_libraries(cv_geometry_utils PUBLIC
  ${OpenCV_LIBS}
  Eigen3::Eigen
  wpimath
)
target_link_libraries(cv_geometry_utils PRIVATE
  apriltag
)

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
