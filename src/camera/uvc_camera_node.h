#pragma once

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <vector>

#include "camera/jpeg_buffer.h"
#include "camera/uvc_camera_config.h"
#include "control_loop/node.h"

#include "libuvc/libuvc.h"

namespace camera {

class UVCCameraNode final : public control_loop::INode {
 public:
  UVCCameraNode(std::string_view output_path, const UVCCameraConfig& config);
  ~UVCCameraNode() override;
  void Start();
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void Callback(const control_loop::Context& context);
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  void RegisterCallback(const std::function<void(const control_loop::Context&)>&
                            callback) override;
  void SetTerminateJpeg(bool terminate_jpeg);

 public:
  void CallBack(uvc_frame_t* frame);  // This should not be used publicly

 private:
  std::string output_path_;
  std::string name_;
  uvc_context_t* context_;
  uvc_device_t* device_;
  uvc_device_handle_t* device_handle_;
  uvc_stream_ctrl_t ctrl_;
  std::unique_ptr<JpegBuffer> buffer_;
  std::atomic<bool> start_ = false;
  std::mutex mutex_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  bool terminate_jpeg_ = true;
};

}  // namespace camera
