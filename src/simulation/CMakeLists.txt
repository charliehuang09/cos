add_library(cos.simulation_position_sender_node)
add_library(cos::simulation_position_sender_node ALIAS cos.simulation_position_sender_node)
target_sources(cos.simulation_position_sender_node
  PRIVATE
    simulation_position_sender_node.cc
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
