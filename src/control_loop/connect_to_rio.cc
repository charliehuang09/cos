#include "control_loop/connect_to_rio.h"

#include "absl/log/log.h"

#include <frc/DataLogManager.h>
#include <networktables/NetworkTableInstance.h>
#include <networktables/StringTopic.h>
#include <ntcore/networktables/NetworkTableInstance.h>
#include <chrono>
#include <filesystem>
#include <thread>

namespace control_loop {
// Publishes logname such as log32 to networktables so we can easily find match logs
static void PublishLogName() {
  std::string path = frc::DataLogManager::GetLogDir();
  static auto log_name_publisher = nt::NetworkTableInstance::GetDefault()
                                       .GetTable("Orin")
                                       ->GetStringTopic("LogName")
                                       .Publish();
  log_name_publisher.Set(path);
}

void StartNetworktables(int team_number) {
  nt::NetworkTableInstance inst = nt::NetworkTableInstance::GetDefault();
  inst.StopServer();
  inst.StopClient();
  inst.StartClient4("orin_localization");
  inst.SetServerTeam(team_number);
  inst.StartDSClient();
  std::string log_path = GetNewLogPath();
  LOG(INFO) << "Log path: " << log_path;
  frc::DataLogManager::Start(log_path);

  LOG(INFO) << "Team number: " << team_number;
  LOG(INFO) << "Waiting for connection";
  while (!inst.IsConnected()) {
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  PublishLogName();
  LOG(INFO) << "Connected to rio!";
}

void StartNetworktablesAsHost() {
  nt::NetworkTableInstance inst = nt::NetworkTableInstance::GetDefault();
  inst.StopLocal();
  inst.StopClient();
  inst.StartServer("orin_localization");
  std::string log_path = GetNewLogPath();
  LOG(INFO) << "Log path: " << log_path;
  frc::DataLogManager::Start(log_path);
}

auto GetNewLogPath(const std::string& log_dir) -> std::string {
  int id = 0;
  std::filesystem::path log_path;
  do {
    log_path = std::filesystem::path(log_dir) / ("log" + std::to_string(id));
    id++;
  } while (std::filesystem::exists(log_path));
  std::filesystem::create_directories(log_path);
  return log_path.string();
}
}  // namespace control_loop
