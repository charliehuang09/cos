#include "apriltag/nvidia_apriltag_detector_node.h"

#include "absl/log/check.h"

#include <vpi/Array.h>
#include <vpi/Context.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <fstream>
#include <vector>

#include <nlohmann/json.hpp>

namespace apriltag {
using control_loop::Context;

namespace {

auto CheckCuda(cudaError_t status) -> void {
  CHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

}  // namespace

static const VPIAprilTagDecodeParams params = {                 // NOLINT
    NULL, 0, 1,                                                 // NOLINT
    VPIAprilTagFamily::VPI_APRILTAG_36H11};                     // NOLINT
static const VPIBackend backend = VPIBackend::VPI_BACKEND_PVA;  // NOLINT
static const int max_detections = 64;

NvidiaApriltagDetectorNode::NvidiaApriltagDetectorNode(
    std::string_view input_channel, std::string_view output_channel,
    std::string_view config_path, control_loop::ThreadPool& thread_pool)
    : input_channel_(input_channel),
      output_channel_(output_channel),
      thread_pool_(thread_pool) {

  std::ifstream config_file{std::string(config_path)};
  CHECK(config_file.is_open()) << "Failed to open config: " << config_path;
  const nlohmann::json config = nlohmann::json::parse(config_file);
  width_ = config.at("width").get<int>();
  height_ = config.at("height").get<int>();
  CHECK(width_ > 0);
  CHECK(height_ > 0);

  CHECK(!vpiContextCreate(backend | VPI_BACKEND_CPU, &context_));
  CHECK(!vpiContextSetCurrent(context_));

  CHECK(
      !vpiCreateAprilTagDetector(backend, width_, height_, &params, &payload_));

  CHECK(!vpiArrayCreate(max_detections, VPI_ARRAY_TYPE_APRILTAG_DETECTION, 0,
                        &detections_));
  // This detector only submits work to PVA. Enabling every stream backend also
  // initializes VPI's CUDA/EGL stack, whose process-exit finalizers conflict
  // with the CUDA context used by nvJPEG on Jetson.
  CHECK(!vpiStreamCreate(backend | VPI_BACKEND_CPU, &stream_));
  CHECK(!vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                        backend | VPI_BACKEND_CPU, &input_));
}

NvidiaApriltagDetectorNode::~NvidiaApriltagDetectorNode() {
  if (stream_ != nullptr) {
    CHECK(!vpiStreamSync(stream_));
    vpiStreamDestroy(stream_);
  }
  if (input_ != nullptr) {
    vpiImageDestroy(input_);
  }
  if (detections_ != nullptr) {
    vpiArrayDestroy(detections_);
  }
  if (payload_ != nullptr) {
    vpiPayloadDestroy(payload_);
  }
  if (context_ != nullptr) {
    vpiContextDestroy(context_);
  }
}

void NvidiaApriltagDetectorNode::WarmUp() {
  std::lock_guard lock(detect_mutex_);
  CHECK(!vpiContextPush(context_));

  VPIImageData image_data{};
  CHECK(!vpiImageLockData(input_, VPI_LOCK_WRITE,
                          VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &image_data));

  auto& plane = image_data.buffer.pitch.planes[0];
  for (int row = 0; row < height_; ++row) {
    std::memset(static_cast<unsigned char*>(plane.pBase) +
                    static_cast<size_t>(row) * plane.pitchBytes,
                0, static_cast<size_t>(width_));
  }
  CHECK(!vpiImageUnlock(input_));

  CHECK(!vpiSubmitAprilTagDetector(stream_, backend, payload_, max_detections,
                                   input_, detections_));
  CHECK(!vpiStreamSync(stream_));

  VPIContext popped_context = nullptr;
  CHECK(!vpiContextPop(&popped_context));
  CHECK(popped_context == context_);
}

auto NvidiaApriltagDetectorNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    Callback(context);
  };
}

void NvidiaApriltagDetectorNode::Callback(const Context& context) {
  auto* decoded_buffer =
      context->GetMessage<camera::DecodedJpegBuffer>(input_channel_);
  if (decoded_buffer == nullptr) [[unlikely]] {
    return;
  }
  std::function<void()> task = [this, context, decoded_buffer]() -> void {
    CHECK(decoded_buffer != nullptr);
    std::unique_ptr<control_loop::IMessage> detections =
        std::make_unique<NvidiaTagDetections>(Detect(*decoded_buffer));
    context->SetMessage(output_channel_, std::move(detections));
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
  thread_pool_.Submit(task);
  return;
}

auto NvidiaApriltagDetectorNode::Detect(const camera::DecodedJpegBuffer& buffer)
    -> std::vector<NvidiaTagDetections::tag_detection> {
  std::lock_guard lock(detect_mutex_);
  CHECK(!vpiContextPush(context_));
  CHECK(buffer.output_format == NVJPEG_OUTPUT_Y);

  VPIImageData image_data{};
  CHECK(!vpiImageLockData(input_, VPI_LOCK_WRITE,
                          VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &image_data));

  auto& plane = image_data.buffer.pitch.planes[0];
  std::vector<unsigned char> gray(buffer.channel_sizes[0]);
  CheckCuda(cudaMemcpy(gray.data(), buffer.destination.channel[0],
                       buffer.channel_sizes[0], cudaMemcpyDeviceToHost));
  for (int row = 0; row < buffer.height; ++row) {
    std::memcpy(static_cast<unsigned char*>(plane.pBase) +
                    static_cast<size_t>(row) * plane.pitchBytes,
                gray.data() + static_cast<size_t>(row) * buffer.stride,
                static_cast<size_t>(buffer.width));
  }
  CHECK(!vpiImageUnlock(input_));

  auto detections = Detect(input_);
  VPIContext popped_context = nullptr;
  CHECK(!vpiContextPop(&popped_context));
  CHECK(popped_context == context_);
  return detections;
}

auto NvidiaApriltagDetectorNode::Detect(VPIImage image)
    -> std::vector<NvidiaTagDetections::tag_detection> {
  CHECK(!vpiSubmitAprilTagDetector(stream_, backend, payload_, max_detections,
                                   image, detections_));

  CHECK(!vpiStreamSync(stream_));

  VPIArrayData detections_data{};
  CHECK(!vpiArrayLockData(detections_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
                          &detections_data));

  auto* detections_vpi =
      static_cast<VPIAprilTagDetection*>(detections_data.buffer.aos.data);
  int num_detections = *detections_data.buffer.aos.sizePointer;

  std::vector<NvidiaTagDetections::tag_detection> detections;

  for (int i = 0; i < num_detections; ++i) {
    NvidiaTagDetections::tag_detection detection;
    detection.tag_id = detections_vpi[i].id;

    for (int j = 0; j < 4; ++j) {
      detection.corners[j] = cv::Point2f(detections_vpi[i].corners[j].x,
                                         detections_vpi[i].corners[j].y);
    }
    detections.push_back(detection);
  }

  CHECK(!vpiArrayUnlock(detections_));

  return detections;
}

}  // namespace apriltag
