#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <thread>

#include "control_loop/rio_clock.h"
#include "networktables/NetworkTableInstance.h"
#include "wpi/timestamp.h"

auto main(int argc, char** argv) -> int {
  using control_loop::RioClock;
  if (!std::isnan(RioClock::GetTime())) {
    std::cerr << "Unsynchronized RioClock must not return a timestamp\n";
    return 1;
  }
  RioClock::EnableSimulation();
  const double simulation_start = RioClock::GetTime();
  std::this_thread::sleep_for(std::chrono::milliseconds{10});
  if (simulation_start < 0 || simulation_start > 1 ||
      RioClock::GetTime() <= simulation_start) {
    std::cerr << "Simulation clock must advance from zero\n";
    return 1;
  }
  RioClock::DisableSimulation();

  constexpr unsigned int kPort = 5810;
  auto server = nt::NetworkTableInstance::Create();
  auto client = nt::NetworkTableInstance::GetDefault();
  // Pass a roboRIO hostname/address to also verify conversion between epochs.
  if (argc == 1) {
    server.StartServer("/tmp/timestamp_test.json", "127.0.0.1",
                       nt::NetworkTableInstance::kDefaultPort3, kPort);
  }
  client.SetServer(argc > 1 ? argv[1] : "127.0.0.1", kPort);
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
  const double rio_time = RioClock::GetTime();
  const double error =
      std::abs(rio_time * 1'000'000.0 - server_time_estimate);
  std::cout << "offset_us=" << *offset << '\n'
            << "server_time_estimate_us=" << server_time_estimate << '\n'
            << "rio_clock_seconds=" << rio_time << '\n'
            << "error_us=" << error << '\n';

  client.StopClient();
  const bool disconnected_time_invalid = std::isnan(RioClock::GetTime());
  if (!disconnected_time_invalid) {
    std::cerr << "Disconnected RioClock must not return a timestamp\n";
  }
  server.StopServer();
  nt::NetworkTableInstance::Destroy(server);

  // RioClock must agree with the synchronized server clock within one millisecond.
  return std::isfinite(rio_time) && error <= 1'000 && disconnected_time_invalid
             ? 0
             : 1;
}
