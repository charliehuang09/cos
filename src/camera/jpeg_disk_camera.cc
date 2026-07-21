#include "camera/jpeg_disk_camera.h"

#include <wpi/system/Timer.hpp>

#include <algorithm>
#include <fstream>

namespace camera {

JpegDiskCamera::JpegDiskCamera(std::string_view folder_path,
                               std::string_view output_channel,
                               size_t queue_length)
    : folder_path_(folder_path),
      output_channel_(output_channel),
      queue_length_(queue_length) {
  std::vector<std::pair<std::filesystem::path, double>> file_paths_vector;
  for (const auto& entry : std::filesystem::directory_iterator(folder_path)) {
    if (entry.is_regular_file()) {
      file_paths_vector.emplace_back(entry.path(),
                                     GetTimestamp(entry.path().filename()));
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

  UpdateJpegBuffer();
}

auto JpegDiskCamera::GetTimestamp(const std::string& filename) -> double {
  return std::stod(filename.substr(0, filename.size() - 4));
}

void JpegDiskCamera::UpdateJpegBuffer() {
  std::lock_guard<std::mutex> gaurd(mutex_);
  while (jpeg_buffers_.size() < queue_length_ && !file_paths_.empty()) {
    std::ifstream file(file_paths_.front().first,
                       std::ios::binary | std::ios::ate);
    JpegBuffer buffer(file.tellg(), file_paths_.front().second);
    file.seekg(0);
    file.read(static_cast<char*>(buffer.ptr), buffer.size);
    file_paths_.pop();
    jpeg_buffers_.push(std::move(buffer));
  }
}

void JpegDiskCamera::Callback(const control_loop::Context& context) {
  std::lock_guard<std::mutex> gaurd(mutex_);
  const double now = wpi::Timer::GetTimestamp().value();
  if (!replay_start_time_.has_value()) {
    replay_start_time_ = now;
  }
  const double replay_time = now - replay_start_time_.value();
  if (!jpeg_buffers_.empty() &&
      jpeg_buffers_.front().timestamp <= replay_time) {
    std::unique_ptr<control_loop::IMessage> message =
        std::make_unique<JpegBuffer>(std::move(jpeg_buffers_.front()));
    jpeg_buffers_.pop();
    context->SetMessage(output_channel_, std::move(message));
  } else {
    context->SetMessage(output_channel_, std::make_unique<JpegBuffer>());
  }
}

auto JpegDiskCamera::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    Callback(context);
    UpdateJpegBuffer();
  };
}
}  // namespace camera
