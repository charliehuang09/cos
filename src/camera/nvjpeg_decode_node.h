#pragma once
#include <array>
#include <cstddef>
#include <mutex>
#include <string>
#include <utility>

#include <nvjpeg.h>

#include "camera/uvc_camera_node.h"
#include "control_loop/message.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"
namespace camera {

class DecodedJpegBuffer final : public control_loop::IMessage {
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
  nvjpegOutputFormat_t output_format = NVJPEG_OUTPUT_Y;
  ;
  std::array<size_t, NVJPEG_MAX_COMPONENT> channel_sizes = {};
  nvjpegImage_t destination = {};
};

class NvjpegDecodeNode final : public control_loop::INode {
 public:
  explicit NvjpegDecodeNode(std::string_view input_path,
                            std::string_view output_path,
                            nvjpegOutputFormat_t output_format,
                            control_loop::ThreadPool& thread_pool);
  ~NvjpegDecodeNode() override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(
      std::function<void(const control_loop::Context&)> callback) override {
    callbacks_.emplace_back(std::move(callback));
  };

 private:
  auto DecodeJpegBuffer(const JpegBuffer* jpeg_buffer) -> DecodedJpegBuffer;

 private:
  std::string input_path_;
  std::string output_path_;
  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
  std::mutex decode_mutex_;
  nvjpegOutputFormat_t output_format_ = NVJPEG_OUTPUT_BGRI;
  control_loop::ThreadPool& thread_pool_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace camera
