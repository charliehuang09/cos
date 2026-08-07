add_library(simulation_position_sender_node)
target_sources(simulation_position_sender_node
  PRIVATE
    simulation_position_sender_node.cc
)
target_link_libraries(simulation_position_sender_node PRIVATE
  absl::log
  ixwebsocket
)

