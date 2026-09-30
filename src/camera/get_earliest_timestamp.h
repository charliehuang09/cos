#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace camera {

auto GetEarliestTimestamp(std::string_view path) -> double;
auto GetEarliestTimestamp(const std::vector<std::string>& paths) -> double;

}  // namespace camera
