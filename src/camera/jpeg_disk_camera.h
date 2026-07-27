#pragma once

#include <filesystem>
#include <functional>
#include <optional>
#include <queue>

#include "control_loop/node.h"

namespace camera {

class JpegDiskCamera final : public control_loop::INode {
 public:
  JpegDiskCamera(std::string_view folder_path, std::string_view output_channel,
                 bool stop_when_empty = true, bool replay_all_frames = false);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  void EnableLogging();

 private:
  void Callback(const control_loop::Context& context);
  auto GetTimestamp(const std::filesystem::path& path) -> std::optional<double>;

 private:
  std::string output_channel_;
  bool stop_when_empty_;
  bool replay_all_frames_;
  std::queue<std::pair<std::filesystem::path, double>> file_paths_;
  std::optional<double> replay_start_time_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  bool logging_ = false;
};

}  // namespace camera
