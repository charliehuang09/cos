#include "control_loop/connect_to_rio.h"
#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"

ABSL_FLAG(int, team_number, 971,  // NOLINT
          "Team number");         // NOLINT
auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);
  control_loop::StartNetworktables(absl::GetFlag(FLAGS_team_number));
}
