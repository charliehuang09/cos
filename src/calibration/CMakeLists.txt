add_library(cos.extrinsics_replay extrinsics_replay.cc)
add_library(cos::extrinsics_replay ALIAS cos.extrinsics_replay)
target_sources(cos.extrinsics_replay PUBLIC FILE_SET HEADERS
  BASE_DIRS ${PROJECT_SOURCE_DIR}/src FILES extrinsics_replay.h)
target_link_libraries(cos.extrinsics_replay PUBLIC
  cos::unambiguous_solver_node cos::thread_pool)

add_library(cos.extrinsics_solver extrinsics_solver.cc)
add_library(cos::extrinsics_solver ALIAS cos.extrinsics_solver)
target_sources(cos.extrinsics_solver PUBLIC FILE_SET HEADERS
  BASE_DIRS ${PROJECT_SOURCE_DIR}/src FILES extrinsics_solver.h)
target_link_libraries(cos.extrinsics_solver PUBLIC cos::extrinsics_replay
  PRIVATE Ceres::ceres)
