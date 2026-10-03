#pragma once

#include <filesystem>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace camera {

// JPEG paths and capture times in seconds, sorted by capture time.
auto GetTimestampedJpegs(const std::filesystem::path& path)
    -> std::vector<std::pair<std::filesystem::path, double>>;
auto GetEarliestTimestamp(std::string_view path) -> double;
auto GetEarliestTimestamp(const std::vector<std::string>& paths) -> double;

}  // namespace camera
