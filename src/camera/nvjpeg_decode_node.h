#pragma once
#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <queue>
#include <string>
#include <thread>
#include <vector>

#include <nvjpeg.h>

#include "camera/uvc_camera_node.h"
namespace camera {

class DecodedJpegBuffer {
 public:
  int width = 0;
  int height = 0;
  size_t stride = 0;
  std::vector<unsigned char> bgr;
};

class NvjpegDecodeNode {
 public:
  NvjpegDecodeNode(const std::string& name);
  ~NvjpegDecodeNode();
  void RegisterCallback(
      const std::function<void(std::shared_ptr<DecodedJpegBuffer>)>&
          callback);
  void Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer);

 private:
  void DecodeJpegBuffer(const std::shared_ptr<JpegBuffer>& jpeg_buffer);

 private:
  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
  std::condition_variable_any cv_;
  std::timed_mutex mutex_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::function<void(std::shared_ptr<DecodedJpegBuffer>)>>
      callbacks_;
  std::jthread decode_thread_;
};

}  // namespace camera
