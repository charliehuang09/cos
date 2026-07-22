#include "camera/nvjpeg_decode_node.h"

#include <array>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace camera {
using std::function;

namespace {

// auto CheckNvjpeg(nvjpegStatus_t status) -> void {
//   CHECK(status == NVJPEG_STATUS_SUCCESS);
// }

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
      timestamp(other.timestamp),
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
    : input_path_({std::string(input_path)}),
      output_path_(output_path),
      output_format_(output_format),
      thread_pool_(thread_pool),
      dependencies_({{input_path_, typeid(JpegBuffer)}}),
      publications_({{output_path_, typeid(DecodedJpegBuffer)}}) {
  CHECK(nvjpegCreateSimple(&handle_) == NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_GPU_HYBRID, &decoder_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecoderStateCreate(handle_, decoder_, &state_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegJpegStreamCreate(handle_, &jpeg_stream_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecodeParamsCreate(handle_, &decode_params_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecodeParamsSetOutputFormat(decode_params_, output_format_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegBufferPinnedCreate(handle_, nullptr, &pinned_buffer_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegBufferDeviceCreate(handle_, nullptr, &device_buffer_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegStateAttachPinnedBuffer(state_, pinned_buffer_) ==
        NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegStateAttachDeviceBuffer(state_, device_buffer_) ==
        NVJPEG_STATUS_SUCCESS);
}

NvjpegDecodeNode::~NvjpegDecodeNode() {
  LOG(INFO) << "Destructing NvjpegDecodeNode";
  if (device_buffer_ != nullptr) {
    CHECK(nvjpegBufferDeviceDestroy(device_buffer_) == NVJPEG_STATUS_SUCCESS);
  }
  if (pinned_buffer_ != nullptr) {
    CHECK(nvjpegBufferPinnedDestroy(pinned_buffer_) == NVJPEG_STATUS_SUCCESS);
  }
  if (decode_params_ != nullptr) {
    CHECK(nvjpegDecodeParamsDestroy(decode_params_) == NVJPEG_STATUS_SUCCESS);
  }
  if (jpeg_stream_ != nullptr) {
    CHECK(nvjpegJpegStreamDestroy(jpeg_stream_) == NVJPEG_STATUS_SUCCESS);
  }
  if (state_ != nullptr) {
    CHECK(nvjpegJpegStateDestroy(state_) == NVJPEG_STATUS_SUCCESS);
  }
  if (decoder_ != nullptr) {
    CHECK(nvjpegDecoderDestroy(decoder_) == NVJPEG_STATUS_SUCCESS);
  }
  if (handle_ != nullptr) {
    CHECK(nvjpegDestroy(handle_) == NVJPEG_STATUS_SUCCESS);
  }
}

auto NvjpegDecodeNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto* jpeg_buffer = context->GetMessage<JpegBuffer>(input_path_);
    if (jpeg_buffer == nullptr || jpeg_buffer->ptr == nullptr) {
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
  std::lock_guard lock(decode_mutex_);

  int components = 0;
  nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_UNKNOWN;
  std::array<int, NVJPEG_MAX_COMPONENT> widths = {};
  std::array<int, NVJPEG_MAX_COMPONENT> heights = {};
  const auto* jpeg_data = static_cast<unsigned char*>(jpeg_buffer->ptr);

  CHECK_EQ(
      nvjpegGetImageInfo(handle_, jpeg_data, jpeg_buffer->size, &components,
                         &subsampling, widths.data(), heights.data()),
      NVJPEG_STATUS_SUCCESS);

  DecodedJpegBuffer decoded_buffer{};
  decoded_buffer.timestamp = jpeg_buffer->timestamp;

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

  CHECK(nvjpegJpegStreamParse(handle_, jpeg_data, jpeg_buffer->size, 0, 0,
                              jpeg_stream_) == NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecodeJpegHost(handle_, decoder_, state_, decode_params_,
                             jpeg_stream_) == NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecodeJpegTransferToDevice(handle_, decoder_, state_,
                                         jpeg_stream_,
                                         nullptr) == NVJPEG_STATUS_SUCCESS);
  CHECK(nvjpegDecodeJpegDevice(handle_, decoder_, state_,
                               &decoded_buffer.destination,
                               nullptr) == NVJPEG_STATUS_SUCCESS);
  CheckCuda(cudaDeviceSynchronize());

  return decoded_buffer;
}

auto NvjpegDecodeNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto NvjpegDecodeNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

}  // namespace camera
