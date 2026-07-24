#include "camera/nvjpeg_fd_decode_node.h"

#include <utility>
#include <vector>

#include "NvBufSurface.h"
#include "NvJpegDecoder.h"
#include "absl/log/check.h"
#include "control_loop/timer.h"
#include "nvbufsurface.h"

namespace camera {

DecodedJpegFdBuffer::~DecodedJpegFdBuffer() {
  if (fd >= 0) {
    CHECK_EQ(NvBufSurf::NvDestroy(fd), 0);
  }
}

DecodedJpegFdBuffer::DecodedJpegFdBuffer(DecodedJpegFdBuffer&& other) noexcept
    : fd(std::exchange(other.fd, -1)),
      pixel_format(other.pixel_format),
      width(other.width),
      height(other.height),
      stride(other.stride),
      output_size(other.output_size),
      timestamp(other.timestamp) {}

NvjpegFdDecodeNode::NvjpegFdDecodeNode(std::string_view input_path,
                                       std::string_view output_path,
                                       control_loop::ThreadPool& thread_pool)
    : input_path_({std::string(input_path)}),
      output_path_(output_path),
      thread_pool_(thread_pool),
      dependencies_({{input_path_, typeid(JpegBuffer)}}),
      publications_({{output_path_, typeid(DecodedJpegFdBuffer)}}) {
  decoder_ = NvJPEGDecoder::createJPEGDecoder("cos-jpeg-fd-decoder");
  CHECK(decoder_ != nullptr);
}

NvjpegFdDecodeNode::~NvjpegFdDecodeNode() {
  delete decoder_;
}

auto NvjpegFdDecodeNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto* jpeg_buffer = context->GetMessage<JpegBuffer>(input_path_);
    if (jpeg_buffer == nullptr || jpeg_buffer->ptr == nullptr) {
      return;
    }

    thread_pool_.Submit([this, context, jpeg_buffer]() -> void {
      control_loop::Timer timer;
      std::unique_ptr<control_loop::IMessage> decoded_buffer =
          std::make_unique<DecodedJpegFdBuffer>(DecodeJpegBuffer(jpeg_buffer));
      context->SetMessage(output_path_, std::move(decoded_buffer));

      if (latency_channel_.has_value()) {
        context->SetMessage(
            latency_channel_.value(),
            std::make_unique<control_loop::LatencyMessage>(timer.Stop()));
      }
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    });
  };
}

auto NvjpegFdDecodeNode::DecodeJpegBuffer(const JpegBuffer* jpeg_buffer)
    -> DecodedJpegFdBuffer {
  std::lock_guard lock(decode_mutex_);

  auto* jpeg_data = static_cast<unsigned char*>(jpeg_buffer->ptr);
  size_t jpeg_size = jpeg_buffer->size;
  std::vector<unsigned char> terminated_jpeg;
  if (jpeg_size < 2U || jpeg_data[jpeg_size - 2U] != 0xFFU ||
      jpeg_data[jpeg_size - 1U] != 0xD9U) {
    terminated_jpeg.assign(jpeg_data, jpeg_data + jpeg_size);
    terminated_jpeg.push_back(0xFFU);
    terminated_jpeg.push_back(0xD9U);
    jpeg_data = terminated_jpeg.data();
    jpeg_size = terminated_jpeg.size();
  }

  int decoded_fd = -1;
  uint32_t pixel_format = 0;
  uint32_t width = 0;
  uint32_t height = 0;
  CHECK_EQ(decoder_->decodeToFd(decoded_fd, jpeg_data, jpeg_size, pixel_format,
                                width, height),
           0);

  NvBufSurface* decoded_surface = nullptr;
  CHECK_EQ(NvBufSurfaceFromFd(decoded_fd,
                              reinterpret_cast<void**>(&decoded_surface)),
           0);
  CHECK(decoded_surface != nullptr);
  const auto& source = decoded_surface->surfaceList[0];

  NvBufSurf::NvCommonAllocateParams allocate_params{};
  allocate_params.width = width;
  allocate_params.height = height;
  allocate_params.layout = NVBUF_LAYOUT_PITCH;
  allocate_params.colorFormat = source.colorFormat;
  allocate_params.memType = NVBUF_MEM_SURFACE_ARRAY;
  allocate_params.memtag = NvBufSurfaceTag_JPEG;

  DecodedJpegFdBuffer output{};
  CHECK_EQ(NvBufSurf::NvAllocate(&allocate_params, 1, &output.fd), 0);
  NvBufSurf::NvCommonTransformParams transform_params{};
  transform_params.src_width = width;
  transform_params.src_height = height;
  transform_params.dst_width = width;
  transform_params.dst_height = height;
  transform_params.flag = static_cast<NvBufSurfTransform_Transform_Flag>(
      NVBUFSURF_TRANSFORM_FILTER | NVBUFSURF_TRANSFORM_CROP_SRC);
  transform_params.flip = NvBufSurfTransform_None;
  transform_params.filter = NvBufSurfTransformInter_Nearest;
  CHECK_EQ(NvBufSurf::NvTransform(&transform_params, decoded_fd, output.fd), 0);
  CHECK_EQ(NvBufSurf::NvDestroy(decoded_fd), 0);

  NvBufSurface* output_surface = nullptr;
  CHECK_EQ(
      NvBufSurfaceFromFd(output.fd, reinterpret_cast<void**>(&output_surface)),
      0);
  CHECK(output_surface != nullptr);
  const auto& destination = output_surface->surfaceList[0];
  output.pixel_format = pixel_format;
  output.width = static_cast<int>(width);
  output.height = static_cast<int>(height);
  output.stride = destination.planeParams.pitch[0];
  for (uint32_t plane = 0; plane < destination.planeParams.num_planes;
       ++plane) {
    output.output_size += destination.planeParams.psize[plane];
  }
  output.timestamp = jpeg_buffer->timestamp;
  return output;
}

auto NvjpegFdDecodeNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto NvjpegFdDecodeNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void NvjpegFdDecodeNode::EnableTiming(std::string_view latency_channel) {
  publications_.emplace_back(latency_channel,
                             typeid(control_loop::LatencyMessage));
  latency_channel_ = latency_channel;
}

}  // namespace camera
