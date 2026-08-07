#include <ixwebsocket/IXNetSystem.h>
#include <ixwebsocket/IXWebSocketServer.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <thread>

namespace {

constexpr int kPort = 5805;
std::atomic_bool keepRunning{true};
std::mutex clientsMutex;
std::set<std::shared_ptr<ix::WebSocket>> clients;

void stop(int) {
  keepRunning = false;
}

auto makePose(double seconds) -> std::string {
  (void)seconds;
  const auto timestamp = std::chrono::duration_cast<std::chrono::milliseconds>(
                             std::chrono::system_clock::now().time_since_epoch())
                             .count();

  std::ostringstream json;
  json.setf(std::ios::fixed);
  json.precision(5);
  json << R"({"position":{"x":0,"y":0,"z":0},)"
       << R"("quaternion":{"x":0,"y":0,"z":0,"w":1},)"
       << "\"timestamp\":" << timestamp << "}";
  return json.str();
}

}  // namespace

auto main() -> int {
  std::signal(SIGINT, stop);
  std::signal(SIGTERM, stop);

  ix::initNetSystem();
  ix::WebSocketServer server(kPort, "127.0.0.1");
  server.setOnConnectionCallback([](const std::weak_ptr<ix::WebSocket>& webSocket,
                                    const std::shared_ptr<ix::ConnectionState>& connectionState) -> void {
    const auto socket = webSocket.lock();
    if (!socket) return;

    socket->setOnMessageCallback([webSocket, connectionState](const ix::WebSocketMessagePtr& message) -> void {
      const auto socket = webSocket.lock();
      if (!socket) return;

      if (message->type == ix::WebSocketMessageType::Open) {
        if (message->openInfo.uri != "/robot") {
          socket->close(1008, "Only /robot is available");
          return;
        }

        std::lock_guard lock(clientsMutex);
        clients.insert(socket);
        std::cout << "Viewer connected from " << connectionState->getRemoteIp() << "\n";
      } else if (message->type == ix::WebSocketMessageType::Close) {
        std::lock_guard lock(clientsMutex);
        clients.erase(socket);
      }
    });
  });

  const auto [listening, error] = server.listen();
  if (!listening) {
    std::cerr << "Unable to listen on ws://127.0.0.1:" << kPort << "/robot: " << error << "\n";
    ix::uninitNetSystem();
    return 1;
  }

  server.start();
  std::cout << "Robot test server listening at ws://127.0.0.1:" << kPort << "/robot\n";

  const auto startedAt = std::chrono::steady_clock::now();
  while (keepRunning) {
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - startedAt).count();
    const std::string pose = makePose(seconds);
    std::set<std::shared_ptr<ix::WebSocket>> connectedClients;
    {
      std::lock_guard lock(clientsMutex);
      connectedClients = clients;
    }
    for (const auto& client : connectedClients) client->send(pose);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }

  server.stop();
  ix::uninitNetSystem();
  std::cout << "Robot test server stopped\n";
}
