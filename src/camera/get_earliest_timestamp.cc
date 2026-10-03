#include "camera/get_earliest_timestamp.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <limits>
#include <stdexcept>

namespace {

auto IsJpeg(const std::filesystem::path& path) -> bool {
  std::string extension = path.extension().string();
  std::ranges::transform(extension, extension.begin(),
                         [](unsigned char character) -> char {
                           return static_cast<char>(std::tolower(character));
                         });
  return extension == ".jpg" || extension == ".jpeg";
}

auto TimestampFromFilename(const std::filesystem::path& path) -> double {
  const std::string stem = path.stem().string();
  try {
    std::size_t parsed_characters = 0;
    const double timestamp = std::stod(stem, &parsed_characters);
    if (parsed_characters == stem.size() && std::isfinite(timestamp)) {
      return timestamp;
    }
  } catch (const std::invalid_argument&) {
  } catch (const std::out_of_range&) {
  }
  return std::numeric_limits<double>::infinity();
}

}  // namespace

namespace camera {

auto GetTimestampedJpegs(const std::filesystem::path& path)
    -> std::vector<std::pair<std::filesystem::path, double>> {
  std::vector<std::pair<std::filesystem::path, double>> files;
  for (const auto& entry : std::filesystem::directory_iterator(path)) {
    if (!entry.is_regular_file() || !IsJpeg(entry.path())) {
      continue;
    }

    const double timestamp = TimestampFromFilename(entry.path());
    if (std::isfinite(timestamp)) files.emplace_back(entry.path(), timestamp);
  }
  std::ranges::sort(files, {}, [](const auto& file) { return file.second; });
  return files;
}

auto GetEarliestTimestamp(std::string_view path) -> double {
  const auto files = GetTimestampedJpegs(std::filesystem::path(path));
  if (files.empty()) {
    throw std::invalid_argument("No JPEG file with a valid timestamp in " +
                                std::string(path));
  }

  return files.front().second;
}

auto GetEarliestTimestamp(const std::vector<std::string>& paths) -> double {
  if (paths.empty()) {
    throw std::invalid_argument("No paths provided");
  }

  double earliest_timestamp = std::numeric_limits<double>::infinity();
  for (const std::string& path : paths) {
    earliest_timestamp =
        std::min(earliest_timestamp, GetEarliestTimestamp(path));
  }
  return earliest_timestamp;
}

}  // namespace camera
