#pragma once

#include <filesystem>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#include "camera/uvc_camera_node.h"
namespace camera {

class UVCDiskCameraNode final : public control_loop::INode {
 public:
  UVCDiskCameraNode(std::string_view log_path, std::string_view output_path,
                    double offset);
  ~UVCDiskCameraNode() override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void Callback(const control_loop::Context& context);
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;

 private:
  std::vector<std::pair<std::filesystem::path, double>> file_paths_;
  std::jthread thread_;
  std::mutex mutex_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::string output_path_;
  std::unique_ptr<JpegBuffer> buffer_ = nullptr;
  bool playback_complete_ = false;
};
}  // namespace camera
