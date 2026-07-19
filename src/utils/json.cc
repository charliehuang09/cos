#include "utils/json.h"

#include <fstream>

#include "absl/log/check.h"

namespace utils {

auto ReadJson(const std::string& path) -> nlohmann::json {
  std::ifstream file(path);
  CHECK(file.is_open()) << "Failed to open json file: " << path;
  return nlohmann::json::parse(file);
}

}  // namespace utils
