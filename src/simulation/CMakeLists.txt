add_library(cos.simulation_position_sender_node)
add_library(cos::simulation_position_sender_node ALIAS cos.simulation_position_sender_node)
target_sources(cos.simulation_position_sender_node
  PRIVATE
    simulation_position_sender_node.cc
  PUBLIC
    FILE_SET HEADERS
    BASE_DIRS ${PROJECT_SOURCE_DIR}/src
    FILES
      simulation_position_sender_node.h
)
target_link_libraries(cos.simulation_position_sender_node
  PRIVATE
    absl::log
    wpilibc
    nlohmann_json::nlohmann_json
  PUBLIC
    ixwebsocket
)
target_compile_options(cos.simulation_position_sender_node PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)
