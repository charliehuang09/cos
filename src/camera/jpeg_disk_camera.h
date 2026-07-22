#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <queue>

#include "control_loop/node.h"

namespace camera {

class JpegDiskCamera final : public control_loop::INode {
 public:
  JpegDiskCamera(std::string_view folder_path, std::string_view output_channel);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  auto GetDependencies()
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  void Callback(const control_loop::Context& context);
  auto GetTimestamp(const std::filesystem::path& path) -> std::optional<double>;

 private:
  std::string output_channel_;
  std::queue<std::pair<std::filesystem::path, double>> file_paths_;
  std::optional<double> replay_start_time_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
};

}  // namespace camera
