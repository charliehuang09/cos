#pragma once

#include <atomic>
#include <cstdlib>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

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
  JpegBuffer() : size(0), timestamp(0), ptr(nullptr) {}
  JpegBuffer(size_t size, double timestamp)
      : size(size), timestamp(timestamp), ptr(std::malloc(size)) {}
  ~JpegBuffer() override { std::free(ptr); }
  JpegBuffer(const JpegBuffer&) = delete;
  JpegBuffer(JpegBuffer&& other) noexcept
      : size(other.size), timestamp(other.timestamp), ptr(other.ptr) {
    other.ptr = nullptr;
  }

  size_t size;
  double timestamp;
  void* ptr;
  auto GetType() -> const std::type_info& override {
    return typeid(JpegBuffer);
  }
  auto GetSize() -> size_t override { return sizeof(*this) + size; }
};

class UVCCameraNode final : public control_loop::INode {
 public:
  UVCCameraNode(std::string_view output_path, const UVCCameraConfig& config);
  ~UVCCameraNode() override;
  void Start();
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void Callback(const control_loop::Context& context);
  void CallBack(uvc_frame_t* frame);  // This should not be used publicly
  auto GetDependencies() -> const std::vector<std::string>& override;

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
  std::vector<std::string> dependencies_;
};

}  // namespace camera
