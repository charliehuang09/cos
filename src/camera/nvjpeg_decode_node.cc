#include "camera/nvjpeg_decode_node.h"

#include <array>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace camera {
namespace {

auto CheckNvjpeg(nvjpegStatus_t status) -> void {
  CHECK(status == NVJPEG_STATUS_SUCCESS);
}

auto CheckCuda(cudaError_t status) -> void {
  CHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

}  // namespace

NvjpegDecodeNode::NvjpegDecodeNode(const std::string& name) {
  (void)name;
  CheckNvjpeg(nvjpegCreateSimple(&handle_));
  CheckNvjpeg(nvjpegJpegStateCreate(handle_, &state_));
  decode_thread_ = std::jthread([this](const std::stop_token& stop_token) {
    std::function<void()> task;
    while (!stop_token.stop_requested()) {
      {
        std::unique_lock<std::timed_mutex> lock(mutex_);
        cv_.wait(lock, [this, stop_token] {
          return !tasks_.empty() || stop_token.stop_requested();
        });
        if (tasks_.empty()) {
          continue;
        }
        task = std::move(tasks_.front());
        tasks_.pop();
      }
      if (stop_token.stop_requested()) {
        break;
      }
      task();
    }
  });
}

NvjpegDecodeNode::~NvjpegDecodeNode() {
  LOG(INFO) << "Destructing NvjpegDecodeNode";
  decode_thread_.request_stop();
  cv_.notify_one();
  if (decode_thread_.joinable()) {
    decode_thread_.join();
  }
  if (state_ != nullptr) {
    CheckNvjpeg(nvjpegJpegStateDestroy(state_));
  }
  if (handle_ != nullptr) {
    CheckNvjpeg(nvjpegDestroy(handle_));
  }
}
void NvjpegDecodeNode::Decode(const std::shared_ptr<JpegBuffer>& jpeg_buffer) {
  std::function<void()> task = [this, jpeg_buffer] {
    DecodeJpegBuffer(jpeg_buffer);
  };
  {
    std::lock_guard<std::timed_mutex> lock(mutex_);
    tasks_.push(task);
    cv_.notify_one();
  }
}

void NvjpegDecodeNode::RegisterCallback(
    const std::function<void(std::shared_ptr<DecodedJpegBuffer>)>& callback) {
  std::unique_lock<std::timed_mutex> lock(mutex_, std::chrono::milliseconds(3));
  callbacks_.push_back(callback);
}

void NvjpegDecodeNode::DecodeJpegBuffer(
    const std::shared_ptr<JpegBuffer>& jpeg_buffer) {
  int components = 0;
  nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
  std::array<int, NVJPEG_MAX_COMPONENT> widths = {};
  std::array<int, NVJPEG_MAX_COMPONENT> heights = {};
  const auto* jpeg_data = static_cast<unsigned char*>(jpeg_buffer->ptr());

  CheckNvjpeg(nvjpegGetImageInfo(handle_, jpeg_data, jpeg_buffer->size(),
                                 &components, &subsampling, widths.data(),
                                 heights.data()));

  auto buffer_shared_ptr = std::make_shared<DecodedJpegBuffer>();
  buffer_shared_ptr->width = widths[0];
  buffer_shared_ptr->height = heights[0];
  buffer_shared_ptr->stride =
      static_cast<size_t>(buffer_shared_ptr->width) * 3;
  buffer_shared_ptr->bgr.resize(buffer_shared_ptr->stride *
                                static_cast<size_t>(buffer_shared_ptr->height));

  unsigned char* device_buffer = nullptr;
  CheckCuda(cudaMalloc(reinterpret_cast<void**>(&device_buffer),
                       buffer_shared_ptr->bgr.size()));

  nvjpegImage_t output = {};
  output.channel[0] = device_buffer;
  output.pitch[0] = buffer_shared_ptr->stride;

  CheckNvjpeg(nvjpegDecode(handle_, state_, jpeg_data, jpeg_buffer->size(),
                           NVJPEG_OUTPUT_BGRI, &output, nullptr));
  CheckCuda(cudaDeviceSynchronize());
  CheckCuda(cudaMemcpy(buffer_shared_ptr->bgr.data(), device_buffer,
                       buffer_shared_ptr->bgr.size(), cudaMemcpyDeviceToHost));
  CheckCuda(cudaFree(device_buffer));

  for (size_t i = 0; i < callbacks_.size(); i++) {  // NOLINT
    std::function<void(std::shared_ptr<DecodedJpegBuffer>)> callback =
        callbacks_[i];
    callback(buffer_shared_ptr);
  }
}

}  // namespace camera
