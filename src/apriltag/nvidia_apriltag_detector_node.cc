#include "apriltag/nvidia_apriltag_detector_node.h"

#include "absl/log/check.h"

#include <vpi/Array.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstring>
#include <vector>

namespace apriltag {
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
    int width, int height, std::string& config_path) {

  CHECK(!vpiCreateAprilTagDetector(backend, width, height, &params, &payload_));

  CHECK(!vpiArrayCreate(max_detections, VPI_ARRAY_TYPE_APRILTAG_DETECTION, 0,
                        &detections_));
  CHECK(!vpiStreamCreate(0, &stream_));
  CHECK(!vpiImageCreate(width, height, VPI_IMAGE_FORMAT_U8,
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
}

void NvidiaApriltagDetectorNode::Detect(
    const std::shared_ptr<camera::DecodedJpegBuffer>& buffer) {
  CHECK(buffer->output_format == NVJPEG_OUTPUT_Y);

  VPIImageData image_data{};
  CHECK(!vpiImageLockData(input_, VPI_LOCK_WRITE,
                          VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &image_data));

  auto& plane = image_data.buffer.pitch.planes[0];
  std::vector<unsigned char> gray(buffer->channel_sizes[0]);
  CheckCuda(cudaMemcpy(gray.data(), buffer->destination.channel[0],
                       buffer->channel_sizes[0], cudaMemcpyDeviceToHost));
  for (int row = 0; row < buffer->height; ++row) {
    std::memcpy(static_cast<unsigned char*>(plane.pBase) +
                    static_cast<size_t>(row) * plane.pitchBytes,
                gray.data() + static_cast<size_t>(row) * buffer->stride,
                static_cast<size_t>(buffer->width));
  }
  CHECK(!vpiImageUnlock(input_));

  Detect(input_);
}

void NvidiaApriltagDetectorNode::Detect(VPIImage image) {
  CHECK(!vpiSubmitAprilTagDetector(stream_, backend, payload_, max_detections,
                                   image, detections_));

  CHECK(!vpiStreamSync(stream_));

  VPIArrayData detections_data{};
  CHECK(!vpiArrayLockData(detections_, VPI_LOCK_READ, VPI_ARRAY_BUFFER_HOST_AOS,
                          &detections_data));

  auto* detections =
      static_cast<VPIAprilTagDetection*>(detections_data.buffer.aos.data);
  int num_detections = *detections_data.buffer.aos.sizePointer;

  auto detections_ptr = std::make_shared<NvidiaTagDetections>();

  for (int i = 0; i < num_detections; ++i) {
    NvidiaTagDetections::tag_detection detection;
    detection.tag_id = detections[i].id;

    for (int j = 0; j < 4; ++j) {
      detection.corners[j] =
          cv::Point2f(detections[i].corners[j].x, detections[i].corners[j].y);
    }
    detections_ptr.get()->tag_detections.push_back(detection);
  }

  CHECK(!vpiArrayUnlock(detections_));

  for (const auto& callback : callbacks_) {
    callback(detections_ptr);
  }
}

void NvidiaApriltagDetectorNode::RegisterCallback(
    const std::function<void(const std::shared_ptr<NvidiaTagDetections>&)>&
        callback) {
  callbacks_.push_back(callback);
}

}  // namespace apriltag
