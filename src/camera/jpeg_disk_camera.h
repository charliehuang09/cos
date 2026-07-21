#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <queue>

#include "control_loop/node.h"

#include "camera/uvc_camera_node.h"

namespace camera {

class JpegDiskCamera final : public control_loop::INode {
 public:
  JpegDiskCamera(std::string_view folder_path, std::string_view output_channel,
                 size_t queue_length);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;

 private:
  void Callback(const control_loop::Context& context);
  auto GetTimestamp(const std::string& filename) -> double;
  void UpdateJpegBuffer();

 private:
  std::string folder_path_;
  std::string output_channel_;
  std::queue<JpegBuffer> jpeg_buffers_;
  std::queue<std::pair<std::filesystem::path, double>> file_paths_;
  size_t queue_length_;
  std::mutex mutex_;
  std::optional<double> replay_start_time_;
};

}  // namespace camera
