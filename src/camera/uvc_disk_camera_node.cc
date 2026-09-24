#include "camera/uvc_disk_camera_node.h"
#include "logging/publication.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "control_loop/rio_clock.h"
#include "utils/stop.h"

namespace camera {

UVCDiskCameraNode::UVCDiskCameraNode(std::string_view log_path,
                                     std::string_view output_path,
                                     double offset, bool stop_on_complete,
                                     bool start_immediately)
    : publications_({control_loop::MessageDescriptor::For<JpegBuffer>(
          output_path)}),
      output_path_(output_path),
      stop_on_complete_(stop_on_complete),
      offset_(offset) {
  for (const auto& entry : std::filesystem::directory_iterator(log_path)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::string extension = entry.path().extension().string();
    std::ranges::transform(extension, extension.begin(),
                           [](unsigned char character) -> char {
                             return static_cast<char>(std::tolower(character));
                           });
    if (extension != ".jpg" && extension != ".jpeg") {
      continue;
    }
    try {
      std::size_t parsed_characters = 0;
      const std::string stem = entry.path().stem().string();
      const double timestamp = std::stod(stem, &parsed_characters);
      if (parsed_characters == stem.size() && std::isfinite(timestamp)) {
        file_paths_.emplace_back(entry.path(), timestamp);
      }
    } catch (const std::invalid_argument&) {
    } catch (const std::out_of_range&) {}
  }
  std::ranges::sort(file_paths_, {},
                    [](const auto& file) -> auto { return file.second; });

  if (start_immediately) {
    StartPlayback();
  }
}

void UVCDiskCameraNode::StartPlayback() {
  CHECK(!thread_.joinable()) << "Disk camera playback already started";
  thread_ = std::jthread([this](const std::stop_token& stop_token) -> void {
    for (std::size_t index = 0;
         index < file_paths_.size() && !stop_token.stop_requested(); ++index) {
      const double replay_timestamp = file_paths_[index].second - offset_;
      while (!stop_token.stop_requested() &&
             control_loop::RioClock::GetTime() < replay_timestamp) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
      if (stop_token.stop_requested()) {
        return;
      }
      std::ifstream file(file_paths_[index].first,
                         std::ios::binary | std::ios::ate);
      if (!file) {
        LOG(WARNING) << "Failed to read file: " << file_paths_[index].first;
        continue;
      }
      const auto size = file.tellg();
      file.seekg(0);
      auto buffer =
          std::make_unique<JpegBuffer>(size, control_loop::RioClock::GetTime());
      file.read(reinterpret_cast<char*>(buffer->ptr), size);
      if (!file) {
        LOG(WARNING) << "Failed to read file: " << file_paths_[index].first;
        continue;
      }
      {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_ = std::move(buffer);
      }
    }
    std::lock_guard<std::mutex> lock(mutex_);
    playback_complete_ = true;
  });
}

UVCDiskCameraNode::~UVCDiskCameraNode() {
  thread_.request_stop();
  if (thread_.joinable()) {
    thread_.join();
  }
}

auto UVCDiskCameraNode::IsPlaybackComplete() -> bool {
  std::lock_guard lock(mutex_);
  return playback_complete_ && buffer_ == nullptr;
}

auto UVCDiskCameraNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    bool request_stop = false;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      if (buffer_ == nullptr) {
        context->include_in_perfomance_metrics = false;
        context->SetMessage(output_path_, nullptr);
        request_stop = playback_complete_;
      } else {
        context->SetMessage(output_path_, std::move(buffer_));
      }
    }
    for (const auto& callback : callbacks_) {
      callback(context);
    }
    if (request_stop && stop_on_complete_) {
      stop::RequestStop();
    }
  };
}

[[nodiscard]] auto UVCDiskCameraNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}
[[nodiscard]] auto UVCDiskCameraNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}
void UVCDiskCameraNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

}  // namespace camera
