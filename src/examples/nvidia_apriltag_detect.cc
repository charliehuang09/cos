#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "absl/log/check.h"
#include "absl/log/globals.h"
#include "absl/log/initialize.h"
#include "absl/log/log.h"

#include <vpi/Array.h>
#include <vpi/Context.h>
#include <vpi/Image.h>
#include <vpi/Stream.h>
#include <vpi/algo/AprilTags.h>

#include <barrier>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

// Reproduces concurrent PVA AprilTag submissions without a camera or JPEG
// decoder. On the dev Orin, run with --serialized=false to exercise the two
// VPI streams concurrently, or --serialized=true to serialize them.
ABSL_FLAG(bool, serialized, false,
          "Serialize both detector submissions with one process-wide mutex.");
ABSL_FLAG(bool, shared_context, false,
          "Create both detectors in one VPI context.");
ABSL_FLAG(int, iterations, 1,
          "Number of simultaneous submissions per detector.");
ABSL_FLAG(int, width, 1280, "Input image width.");
ABSL_FLAG(int, height, 720, "Input image height.");

namespace {

constexpr VPIBackend kBackend = VPI_BACKEND_PVA;
constexpr int kMaxDetections = 64;

std::mutex serialized_detect_mutex;

class Detector final {
 public:
  Detector(int width, int height, VPIContext shared_context = nullptr)
      : width_(width), height_(height), context_(shared_context) {
    CHECK_GT(width_, 0);
    CHECK_GT(height_, 0);

    if (context_ == nullptr) {
      CHECK_EQ(vpiContextCreate(kBackend | VPI_BACKEND_CPU, &context_),
               VPI_SUCCESS);
      owns_context_ = true;
    }
    CHECK_EQ(vpiContextPush(context_), VPI_SUCCESS);

    const VPIAprilTagDecodeParams params = {
        nullptr, 0, 1, VPIAprilTagFamily::VPI_APRILTAG_36H11};
    CHECK_EQ(vpiCreateAprilTagDetector(kBackend, width_, height_, &params,
                                       &payload_),
             VPI_SUCCESS);
    CHECK_EQ(vpiArrayCreate(kMaxDetections, VPI_ARRAY_TYPE_APRILTAG_DETECTION,
                            0, &detections_),
             VPI_SUCCESS);
    CHECK_EQ(vpiStreamCreate(kBackend | VPI_BACKEND_CPU, &stream_),
             VPI_SUCCESS);
    CHECK_EQ(vpiImageCreate(width_, height_, VPI_IMAGE_FORMAT_U8,
                            kBackend | VPI_BACKEND_CPU, &input_),
             VPI_SUCCESS);

    VPIContext popped_context = nullptr;
    CHECK_EQ(vpiContextPop(&popped_context), VPI_SUCCESS);
    CHECK_EQ(popped_context, context_);
  }

  Detector(const Detector&) = delete;
  auto operator=(const Detector&) -> Detector& = delete;

  ~Detector() {
    CHECK_EQ(vpiContextPush(context_), VPI_SUCCESS);
    if (stream_ != nullptr) {
      CHECK_EQ(vpiStreamSync(stream_), VPI_SUCCESS);
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

    VPIContext popped_context = nullptr;
    CHECK_EQ(vpiContextPop(&popped_context), VPI_SUCCESS);
    CHECK_EQ(popped_context, context_);
    if (owns_context_) {
      vpiContextDestroy(context_);
    }
  }

  auto Detect() -> void {
    CHECK_EQ(vpiContextPush(context_), VPI_SUCCESS);

    VPIImageData image_data{};
    CHECK_EQ(vpiImageLockData(input_, VPI_LOCK_WRITE,
                              VPI_IMAGE_BUFFER_HOST_PITCH_LINEAR, &image_data),
             VPI_SUCCESS);
    const auto& plane = image_data.buffer.pitch.planes[0];
    for (int row = 0; row < height_; ++row) {
      std::memset(static_cast<unsigned char*>(plane.pBase) +
                      static_cast<size_t>(row) * plane.pitchBytes,
                  0, static_cast<size_t>(width_));
    }
    CHECK_EQ(vpiImageUnlock(input_), VPI_SUCCESS);

    CHECK_EQ(vpiSubmitAprilTagDetector(stream_, kBackend, payload_,
                                        kMaxDetections, input_, detections_),
             VPI_SUCCESS);
    CHECK_EQ(vpiStreamSync(stream_), VPI_SUCCESS);

    VPIContext popped_context = nullptr;
    CHECK_EQ(vpiContextPop(&popped_context), VPI_SUCCESS);
    CHECK_EQ(popped_context, context_);
  }

 private:
  int width_;
  int height_;
  VPIContext context_ = nullptr;
  VPIPayload payload_ = nullptr;
  VPIArray detections_ = nullptr;
  VPIStream stream_ = nullptr;
  VPIImage input_ = nullptr;
  bool owns_context_ = false;
};

auto RunDetector(Detector& detector, std::barrier<>& start, int iterations,
                 bool serialized) -> void {
  for (int iteration = 0; iteration < iterations; ++iteration) {
    start.arrive_and_wait();
    if (serialized) {
      std::lock_guard lock(serialized_detect_mutex);
      detector.Detect();
    } else {
      detector.Detect();
    }
  }
}

}  // namespace

auto main(int argc, char** argv) -> int {
  absl::ParseCommandLine(argc, argv);
  absl::InitializeLog();
  absl::SetStderrThreshold(absl::LogSeverityAtLeast::kInfo);

  const int iterations = absl::GetFlag(FLAGS_iterations);
  CHECK_GT(iterations, 0);
  const bool serialized = absl::GetFlag(FLAGS_serialized);
  const bool shared_context_enabled = absl::GetFlag(FLAGS_shared_context);
  VPIContext shared_context = nullptr;
  if (shared_context_enabled) {
    CHECK_EQ(vpiContextCreate(kBackend | VPI_BACKEND_CPU, &shared_context),
             VPI_SUCCESS);
  }

  const auto begin = std::chrono::steady_clock::now();
  {
    Detector first(absl::GetFlag(FLAGS_width), absl::GetFlag(FLAGS_height),
                   shared_context);
    Detector second(absl::GetFlag(FLAGS_width), absl::GetFlag(FLAGS_height),
                    shared_context);
    std::barrier start(2);

    std::thread first_thread(RunDetector, std::ref(first), std::ref(start),
                             iterations, serialized);
    std::thread second_thread(RunDetector, std::ref(second), std::ref(start),
                              iterations, serialized);
    first_thread.join();
    second_thread.join();
  }
  const std::chrono::duration<double> elapsed =
      std::chrono::steady_clock::now() - begin;

  if (shared_context != nullptr) {
    vpiContextDestroy(shared_context);
  }

  LOG(INFO) << "Completed " << iterations
            << " concurrent PVA AprilTag submissions per detector"
            << (serialized ? " with serialization" : " without serialization")
            << (shared_context_enabled ? " in one shared context" :
                                         " in separate contexts")
            << " in " << elapsed.count() << " seconds.";
  return 0;
}
