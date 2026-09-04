#pragma once
#include <string>
namespace control_loop {
// Sets up logging and networktables. Should be called at the beggining of every robot's main
void StartNetworktables(int team_number = 971);
void StartNetworktablesAsHost();
auto GetNewLogPath(const std::string& log_dir = "/bos/logs") -> std::string;
}  // namespace control_loop
