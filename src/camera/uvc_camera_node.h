#pragma once

#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "control_loop/control_loop.h"
#include "control_loop/node.h"

#include "libuvc/libuvc.h"

namespace camera {

struct UVCCameraConfig {
  UVCCameraConfig(const std::string& path);
  std::string name;                      // For debugging
  std::optional<std::string> serial_id;  // Used to find which camera to use
  int height;
  int width;
  int fps;
  int max_payload_size = 3072;
  int max_frame_size = 2048589;
};

// Default
// bmHint: 0001
// bFormatIndex: 1
// bFrameIndex: 1
// dwFrameInterval: 83333
// wKeyFrameRate: 0
// wPFrameRate: 0
// wCompQuality: 0
// wCompWindowSize: 0
// wDelay: 0
// dwMaxVideoFrameSize: 2048589
// dwMaxPayloadTransferSize: 3072
// bInterfaceNumber: 1

class JpegBuffer final : public control_loop::IMessage {
 public:
  JpegBuffer(size_t size) : size(size), ptr(std::malloc(size)) {}
  ~JpegBuffer() override { std::free(ptr); }
  size_t size;
  void* ptr;
  auto GetType() -> const std::type_info& override {
    return typeid(JpegBuffer);
  }
};

class UVCCameraNode final : public control_loop::INode {
 public:
  UVCCameraNode(std::string_view output_path, const UVCCameraConfig& config);
  ~UVCCameraNode() override;
  void Start();
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(
      std::function<void(const control_loop::Context&)> callback) override {
    callbacks_.emplace_back(std::move(callback));
  }
  void Callback(const control_loop::Context& context);
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
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace camera
