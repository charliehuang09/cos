#pragma once
#include <string>
namespace control_loop {
// Sets up logging and networktables. Should be called at the beggining of every robot's main
void StartNetworktables(int team_number = 971);
void StartNetworktablesAsHost();
// Creates the log directory on first use and returns the same path thereafter.
auto GetLogPath() -> const std::string&;
auto GetNewLogPath(const std::string& log_dir = "/cos/logs") -> std::string;
}  // namespace control_loop
