#pragma once

#include <set>
#include "control_loop/node.h"
#include "ixwebsocket/IXWebSocket.h"
#include "ixwebsocket/IXWebSocketServer.h"

namespace simulation {

class SimulationPositionSenderNode final : public control_loop::INode {
 public:
  SimulationPositionSenderNode(std::string_view input_path);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  ~SimulationPositionSenderNode() override;

 private:
  std::string input_path_;
  ix::WebSocketServer server_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::set<std::shared_ptr<ix::WebSocket>> clients_;
  std::mutex mutex_;
};

}  // namespace simulation
