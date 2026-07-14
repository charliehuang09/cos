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

DecodedJpegBuffer::DecodedJpegBuffer(DecodedJpegBuffer&& other) noexcept
    : width(other.width),
      height(other.height),
      stride(other.stride),
      output_size(other.output_size),
      output_format(other.output_format),
      channel_sizes(other.channel_sizes),
      destination(other.destination) {
  other.channel_sizes = {};
  other.destination = {};
  other.output_size = 0;
}

NvjpegDecodeNode::NvjpegDecodeNode(std::string_view input_path,
                                   std::string_view output_path,
                                   nvjpegOutputFormat_t output_format,
                                   control_loop::ThreadPool& thread_pool)
    : input_path_(input_path),
      output_path_(output_path),
      output_format_(output_format),
      thread_pool_(thread_pool) {
  CheckNvjpeg(nvjpegCreateSimple(&handle_));
  CheckNvjpeg(nvjpegJpegStateCreate(handle_, &state_));
}

NvjpegDecodeNode::~NvjpegDecodeNode() {
  LOG(INFO) << "Destructing NvjpegDecodeNode";
  if (state_ != nullptr) {
    CheckNvjpeg(nvjpegJpegStateDestroy(state_));
  }
  if (handle_ != nullptr) {
    CheckNvjpeg(nvjpegDestroy(handle_));
  }
}

auto NvjpegDecodeNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    JpegBuffer* jpeg_buffer = context->GetMessage<JpegBuffer>(input_path_);
    if (jpeg_buffer == nullptr) {
      return;
    }

    std::function<void()> task = [this, context, jpeg_buffer]() -> void {
      std::unique_ptr<control_loop::IMessage> decoded_buffer =
          std::make_unique<DecodedJpegBuffer>(DecodeJpegBuffer(jpeg_buffer));

      context->SetMessage(output_path_, std::move(decoded_buffer));

      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };

    thread_pool_.Submit(task);
  };
}

auto NvjpegDecodeNode::DecodeJpegBuffer(const JpegBuffer* const jpeg_buffer)
    -> DecodedJpegBuffer {
  int components = 0;
  nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
  std::array<int, NVJPEG_MAX_COMPONENT> widths = {};
  std::array<int, NVJPEG_MAX_COMPONENT> heights = {};
  const auto* jpeg_data = static_cast<unsigned char*>(jpeg_buffer->ptr);

  CheckNvjpeg(nvjpegGetImageInfo(handle_, jpeg_data, jpeg_buffer->size,
                                 &components, &subsampling, widths.data(),
                                 heights.data()));

  DecodedJpegBuffer decoded_buffer{};

  ConfigureDestination(&decoded_buffer, output_format_, components, widths,
                       heights);

  for (int channel = 0; channel < NVJPEG_MAX_COMPONENT; ++channel) {
    if (decoded_buffer.channel_sizes[channel] == 0U) {
      continue;
    }
    CheckCuda(cudaMalloc(
        reinterpret_cast<void**>(&decoded_buffer.destination.channel[channel]),
        decoded_buffer.channel_sizes[channel]));
  }

  CheckNvjpeg(nvjpegDecode(handle_, state_, jpeg_data, jpeg_buffer->size,
                           output_format_, &decoded_buffer.destination,
                           nullptr));
  CheckCuda(cudaDeviceSynchronize());

  return decoded_buffer;
}

}  // namespace camera
