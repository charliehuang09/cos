#pragma once
#include <array>
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
  ~DecodedJpegBuffer();
  DecodedJpegBuffer() = default;
  DecodedJpegBuffer(const DecodedJpegBuffer&) = delete;
  auto operator=(const DecodedJpegBuffer&) -> DecodedJpegBuffer& = delete;

  int width = 0;
  int height = 0;
  size_t stride = 0;
  size_t output_size = 0;
  nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_BGRI;
  std::array<size_t, NVJPEG_MAX_COMPONENT> channel_sizes = {};
  nvjpegImage_t destination = {};
};

class NvjpegDecodeNode {
 public:
  explicit NvjpegDecodeNode(
      const std::string& name,
      nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_BGRI);
  ~NvjpegDecodeNode();
  void RegisterCallback(
      const std::function<void(std::shared_ptr<DecodedJpegBuffer>)>& callback);
  void Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer);

 private:
  void DecodeJpegBuffer(const std::shared_ptr<JpegBuffer>& jpeg_buffer);

 private:
  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
  nvjpegOutputFormat_t output_format_ = NVJPEG_OUTPUT_BGRI;
  std::condition_variable_any cv_;
  std::timed_mutex mutex_;
  std::queue<std::function<void()>> tasks_;
  std::vector<std::function<void(std::shared_ptr<DecodedJpegBuffer>)>>
      callbacks_;
  std::jthread decode_thread_;
};

}  // namespace camera
