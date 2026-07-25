#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <thread>

#include "networktables/NetworkTableInstance.h"
#include "wpi/timestamp.h"

auto main() -> int {
  constexpr unsigned int kPort = 5810;
  auto server = nt::NetworkTableInstance::Create();
  auto client = nt::NetworkTableInstance::Create();
  server.StartServer("/tmp/timestamp_test.json", "127.0.0.1",
                     nt::NetworkTableInstance::kDefaultPort3, kPort);
  client.SetServer("127.0.0.1", kPort);
  client.StartClient4("timestamp_test");

  std::optional<int64_t> offset;
  for (int i = 0; i < 500 && !offset; ++i) {
    offset = client.GetServerTimeOffset();
    std::this_thread::sleep_for(std::chrono::milliseconds{10});
  }

  if (!offset) {
    std::cerr << "NetworkTables time synchronization did not complete\n";
    return 1;
  }

  const int64_t client_time = wpi::Now();
  const int64_t server_time_estimate = client_time + *offset;
  const int64_t server_time = wpi::Now();
  const int64_t error = std::llabs(server_time_estimate - server_time);
  std::cout << "offset_us=" << *offset << '\n'
            << "server_time_estimate_us=" << server_time_estimate << '\n'
            << "server_time_us=" << server_time << '\n'
            << "error_us=" << error << '\n';

  client.StopClient();
  server.StopServer();
  nt::NetworkTableInstance::Destroy(client);
  nt::NetworkTableInstance::Destroy(server);

  // Loopback should be within one millisecond; remote peers will include RTT jitter.
  return error <= 1'000 ? 0 : 1;
}
