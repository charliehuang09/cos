#pragma once

#include <string>

#include <nlohmann/json.hpp>

namespace utils {

auto ReadJson(const std::string& path) -> nlohmann::json;

}  // namespace utils
