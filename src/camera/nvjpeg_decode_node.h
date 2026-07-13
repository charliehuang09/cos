#pragma once
#include <array>
#include <cstddef>
#include <string>

#include <nvjpeg.h>

#include "camera/uvc_camera_node.h"
#include "control_loop/message.h"
namespace camera {

class DecodedJpegBuffer : public control_loop::IMessage {
 public:
  ~DecodedJpegBuffer() override;
  DecodedJpegBuffer() = default;

  DecodedJpegBuffer(DecodedJpegBuffer&& other) noexcept;

  auto GetType() -> const std::type_info& override {
    return typeid(DecodedJpegBuffer);
  }

  int width = 0;
  int height = 0;
  size_t stride = 0;
  size_t output_size = 0;
  nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_BGRI;
  std::array<size_t, NVJPEG_MAX_COMPONENT> channel_sizes = {};
  nvjpegImage_t destination = {};
};

// using DecodedJpegBuffer = std::shared_ptr<DecodedJpegBufferInternal>;

class NvjpegDecodeNode {
 public:
  explicit NvjpegDecodeNode(
      std::string_view input_path, std::string_view output_path,
      nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_BGRI);
  ~NvjpegDecodeNode();
  auto CreateCallback() -> std::function<void(const control_loop::Context&)>;

 private:
  auto DecodeJpegBuffer(const JpegBuffer* jpeg_buffer) -> DecodedJpegBuffer;

 private:
  std::string input_path_;
  std::string output_path_;
  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
  nvjpegOutputFormat_t output_format_ = NVJPEG_OUTPUT_BGRI;
};

}  // namespace camera
