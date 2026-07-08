#include "camera/nvjpeg_decode_node.h"

#include <array>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace camera {
using std::function;

namespace {

auto CheckNvjpeg(nvjpegStatus_t status) -> void {
  CHECK(status == NVJPEG_STATUS_SUCCESS);
}

auto CheckCuda(cudaError_t status) -> void {
  CHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

auto ConfigureDestination(DecodedJpegBuffer* buffer,
                          nvjpegOutputFormat_t output_format, int components,
                          const std::array<int, NVJPEG_MAX_COMPONENT>& widths,
                          const std::array<int, NVJPEG_MAX_COMPONENT>& heights)
    -> void {
  buffer->output_format = output_format;
  buffer->width = widths[0];
  buffer->height = heights[0];

  auto set_channel = [buffer](int channel, size_t pitch, int height) -> void {
    buffer->destination.pitch[channel] = pitch;
    buffer->channel_sizes[channel] = pitch * static_cast<size_t>(height);
    buffer->output_size += buffer->channel_sizes[channel];
  };

  switch (output_format) {
    case NVJPEG_OUTPUT_RGBI:
    case NVJPEG_OUTPUT_BGRI:
      set_channel(0, static_cast<size_t>(widths[0]) * 3U, heights[0]);
      break;
    case NVJPEG_OUTPUT_RGB:
    case NVJPEG_OUTPUT_BGR:
      for (int channel = 0; channel < 3; ++channel) {
        set_channel(channel, static_cast<size_t>(widths[0]), heights[0]);
      }
      break;
    case NVJPEG_OUTPUT_Y:
      set_channel(0, static_cast<size_t>(widths[0]), heights[0]);
      break;
    case NVJPEG_OUTPUT_YUV:
    case NVJPEG_OUTPUT_UNCHANGED:
      for (int channel = 0; channel < components; ++channel) {
        set_channel(channel, static_cast<size_t>(widths[channel]),
                    heights[channel]);
      }
      break;
    case NVJPEG_OUTPUT_NV12:
      set_channel(0, static_cast<size_t>(widths[0]), heights[0]);
      set_channel(1, static_cast<size_t>(widths[1]) * 2U, heights[1]);
      break;
    case NVJPEG_OUTPUT_YUY2:
      set_channel(0, static_cast<size_t>(widths[0]) * 2U, heights[0]);
      break;
    case NVJPEG_OUTPUT_UNCHANGEDI:
      set_channel(
          0, static_cast<size_t>(widths[0]) * static_cast<size_t>(components),
          heights[0]);
      break;
    case NVJPEG_OUTPUT_UNCHANGEDI_U16:
      set_channel(0,
                  static_cast<size_t>(widths[0]) *
                      static_cast<size_t>(components) * sizeof(unsigned short),
                  heights[0]);
      break;
    default:
      LOG(FATAL) << "Unsupported nvJPEG output format: " << output_format;
  }

  buffer->stride = buffer->destination.pitch[0];
}

}  // namespace

DecodedJpegBuffer::~DecodedJpegBuffer() {
  for (int channel = 0; channel < NVJPEG_MAX_COMPONENT; ++channel) {
    if (channel_sizes[channel] != 0U &&
        destination.channel[channel] != nullptr) {
      cudaError_t status = cudaFree(destination.channel[channel]);
      if (status != cudaSuccess) {
        LOG(ERROR) << cudaGetErrorString(status);
      }
    }
  }
}

NvjpegDecodeNode::NvjpegDecodeNode(const std::string& name,
                                   nvjpegOutputFormat_t output_format)
    : output_format_(output_format) {
  (void)name;
  CheckNvjpeg(nvjpegCreateSimple(&handle_));
  CheckNvjpeg(nvjpegJpegStateCreate(handle_, &state_));
  decode_thread_ =
      std::jthread([this](const std::stop_token& stop_token) -> void {
        function<void()> task;
        while (!stop_token.stop_requested()) {
          {
            std::unique_lock<std::timed_mutex> lock(mutex_);
            cv_.wait(lock, [this, stop_token]() -> bool {
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
  std::function<void()> task = [this, jpeg_buffer]() -> void {
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
  ConfigureDestination(buffer_shared_ptr.get(), output_format_, components,
                       widths, heights);

  for (int channel = 0; channel < NVJPEG_MAX_COMPONENT; ++channel) {
    if (buffer_shared_ptr->channel_sizes[channel] == 0U) {
      continue;
    }
    CheckCuda(cudaMalloc(reinterpret_cast<void**>(
                             &buffer_shared_ptr->destination.channel[channel]),
                         buffer_shared_ptr->channel_sizes[channel]));
  }

  CheckNvjpeg(nvjpegDecode(handle_, state_, jpeg_data, jpeg_buffer->size(),
                           output_format_, &buffer_shared_ptr->destination,
                           nullptr));
  CheckCuda(cudaDeviceSynchronize());

  for (const auto& callback : callbacks_) {
    callback(buffer_shared_ptr);
  }
}

}  // namespace camera
