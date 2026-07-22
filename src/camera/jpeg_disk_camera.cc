#include "camera/jpeg_disk_camera.h"
#include "camera/uvc_camera_node.h"

#include <wpi/system/Timer.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include "absl/log/check.h"
#include "utils/stop.h"

namespace camera {

JpegDiskCamera::JpegDiskCamera(std::string_view folder_path,
                               std::string_view output_channel)
    : output_channel_(output_channel) {
  std::vector<std::pair<std::filesystem::path, double>> file_paths_vector;
  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    std::string extension = entry.path().extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char c) -> char {
                             return static_cast<char>(std::tolower(c));
                           });
    if (extension != ".jpg" && extension != ".jpeg") {
      continue;
    }

    if (const auto timestamp = GetTimestamp(entry.path());
        timestamp.has_value()) {
      file_paths_vector.emplace_back(entry.path(), timestamp.value());
    }
  }

  std::ranges::sort(file_paths_vector, {},
                    [](const auto& file) -> auto { return file.second; });

  if (!file_paths_vector.empty()) {
    const double first_timestamp = file_paths_vector.front().second;
    for (const auto& [file, timestamp] : file_paths_vector) {
      file_paths_.emplace(file, timestamp - first_timestamp);
    }
  }
}

auto JpegDiskCamera::GetTimestamp(const std::filesystem::path& path)
    -> std::optional<double> {
  const std::string stem = path.stem().string();
  try {
    size_t parsed_characters = 0;
    const double timestamp = std::stod(stem, &parsed_characters);
    if (parsed_characters == stem.size() && std::isfinite(timestamp)) {
      return timestamp;
    }
  } catch (const std::invalid_argument&) {
  } catch (const std::out_of_range&) {}
  return std::nullopt;
}

void JpegDiskCamera::Callback(const control_loop::Context& context) {
  const double now = wpi::Timer::GetTimestamp().value();
  if (!replay_start_time_.has_value()) {
    replay_start_time_ = now;
  }
  const double replay_time = now - replay_start_time_.value();
  std::optional<std::pair<std::filesystem::path, double>> image;
  while (!file_paths_.empty() && file_paths_.front().second <= replay_time) {
    image = std::move(file_paths_.front());
    file_paths_.pop();
  }

  if (!image.has_value()) {
    context->SetMessage(output_channel_, std::make_unique<JpegBuffer>());
    if (file_paths_.empty()) {
      stop::stop = true;
    }
    return;
  }

  std::ifstream file(image->first, std::ios::binary | std::ios::ate);
  const std::streampos file_size = file.tellg();
  CHECK(file_size != std::streampos(-1));
  auto buffer = std::make_unique<JpegBuffer>(file_size, image->second);
  file.seekg(0);
  file.read(static_cast<char*>(buffer->ptr), buffer->size);
  context->SetMessage(output_channel_, std::move(buffer));
}

auto JpegDiskCamera::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    Callback(context);
  };
}

auto JpegDiskCamera::GetDependencies()
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}
}  // namespace camera
