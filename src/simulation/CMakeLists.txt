add_library(simulation_position_sender_node)
target_sources(simulation_position_sender_node
  PRIVATE
    simulation_position_sender_node.cc
)
target_link_libraries(simulation_position_sender_node 
  PRIVATE
    absl::log
    wpilibc
    nlohmann_json::nlohmann_json
  PUBLIC
    ixwebsocket
)
target_compile_options(simulation_position_sender_node PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)

