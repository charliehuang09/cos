#include "apriltag/gpu_apriltag_detector.h"
#include "control_loop/timer.h"

#include <tag36h11.h>
#include <cstring>
#include <limits>

#include "absl/log/check.h"
#include "absl/log/log.h"

#include <cuda/std/algorithm>

#define CUDA_CHECK(call)                                          \
  do {                                                            \
    const cudaError_t cuda_check_error = (call);                  \
    if (cuda_check_error != cudaSuccess) {                        \
      LOG(FATAL) << cudaGetErrorString(cuda_check_error) << '\n'; \
    }                                                             \
  } while (0)

namespace apriltag {
GPUApriltagDetector::GPUApriltagDetector(int width, int height,
                                         std::vector<int> target_tag_ids)
    : width_(width),
      height_(height),
      target_tag_ids_(std::move(target_tag_ids)),
      family_(tag36h11_create()) {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  const int stride = width;
  CHECK(width % 4 == 0);
  CHECK(height % 4 == 0);
  CHECK_GT(width, 0);
  CHECK_GT(height, 0);
  CHECK_LE(static_cast<size_t>(width) * height,
           static_cast<size_t>(std::numeric_limits<int>::max() - 255));
  uint8_t* input_buffer = nullptr;
  CUDA_CHECK(cudaHostAlloc(&input_buffer, static_cast<size_t>(width) * height,
                           cudaHostAllocMapped));
  input_view_ = ImageView{.data = input_buffer, .stride = width,
                          .height = height, .width = width};
  input_view_.EnableGpu();

  uint8_t* max_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&max_buffer),
                           width * height / 16, cudaHostAllocMapped));
  uint8_t* min_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&min_buffer),
                           width * height / 16, cudaHostAllocMapped));
  max_view_ = ImageView{.data = max_buffer,
                        .stride = stride / 4,
                        .height = height / 4,
                        .width = width / 4};
  min_view_ = ImageView{.data = min_buffer,
                        .stride = stride / 4,
                        .height = height / 4,
                        .width = width / 4};
  max_view_.EnableGpu();
  min_view_.EnableGpu();

  uint8_t* threshold_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&threshold_buffer),
                           width * height / 16, cudaHostAllocMapped));
  threshold_view_ = ImageView{.data = threshold_buffer,
                              .stride = stride / 4,
                              .height = height / 4,
                              .width = width / 4};
  uint8_t* valid_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&valid_buffer),
                           width * height / 16, cudaHostAllocMapped));
  valid_view_ = ImageView{.data = valid_buffer,
                          .stride = stride / 4,
                          .height = height / 4,
                          .width = width / 4};
  threshold_view_.EnableGpu();
  valid_view_.EnableGpu();

  auto* binarized_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width * height, sizeof(uint8_t)));
  binarized_apriltag_view_ = ImageView{.data = binarized_apriltag_buffer,
                                       .stride = stride,
                                       .height = height,
                                       .width = width};

  auto* segmented_apriltag_buffer =
      static_cast<uint32_t*>(calloc(width * height, sizeof(uint32_t)));
  segmented_apriltag_view_ = ImageView32{.data = segmented_apriltag_buffer,
                                         .stride = stride,
                                         .height = height,
                                         .width = width};

  auto* boundary_segmented_apriltag_buffer =
      static_cast<uint32_t*>(calloc(width * height, sizeof(uint32_t)));
  boundary_segmented_apriltag_view_ =
      ImageView32{.data = boundary_segmented_apriltag_buffer,
                  .stride = stride,
                  .height = height,
                  .width = width};

  auto* sorted_boundary_segmented_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width * height, sizeof(uint8_t)));
  sorted_boundary_segmented_apriltag_view_ =
      ImageView{.data = sorted_boundary_segmented_apriltag_buffer,
                .stride = stride,
                .height = height,
                .width = width};

  auto* candidate_quad_corners_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width * height, sizeof(uint8_t)));
  candidate_quad_corners_apriltag_view_ =
      ImageView{.data = candidate_quad_corners_apriltag_buffer,
                .stride = stride,
                .height = height,
                .width = width};

  auto* quad_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width * height, sizeof(uint8_t)));
  quad_apriltag_view_ = ImageView{.data = quad_apriltag_buffer,
                                  .stride = stride,
                                  .height = height,
                                  .width = width};

  auto* bit_locations_apriltag_buffer =
      static_cast<uint32_t*>(calloc(width * height, sizeof(uint32_t)));
  bit_locations_apriltag_view_ = ImageView32{
      .data = bit_locations_apriltag_buffer,
      .stride = stride,
      .height = height,
      .width = width,
  };

  auto* refined_points_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width * height, sizeof(uint8_t)));
  refined_points_apriltag_view_ =
      ImageView{.data = refined_points_apriltag_buffer,
                .stride = stride,
                .height = height,
                .width = width};

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_image_),
                        width * height * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_labels_),
                        width * height * sizeof(uint32_t)));
  tag_decoder_.SetTargetCodes(family_, target_tag_ids_);
}
GPUApriltagDetector::~GPUApriltagDetector() {
  cudaFreeHost(input_view_.data);
  cudaFree(device_image_);
  cudaFree(device_labels_);
  cudaFreeHost(max_view_.data);
  cudaFreeHost(min_view_.data);
  cudaFreeHost(threshold_view_.data);
  cudaFreeHost(valid_view_.data);
  free(binarized_apriltag_view_.data);
  free(segmented_apriltag_view_.data);
  free(boundary_segmented_apriltag_view_.data);
  free(sorted_boundary_segmented_apriltag_view_.data);
  free(candidate_quad_corners_apriltag_view_.data);
  free(quad_apriltag_view_.data);
  free(bit_locations_apriltag_view_.data);
  free(refined_points_apriltag_view_.data);
  tag36h11_destroy(family_);
}
auto GPUApriltagDetector::Detect(ImageView apriltag, bool generate_debug_image)
    -> std::vector<ApriltagDetection> {
  CHECK_EQ(apriltag.width, width_);
  CHECK_EQ(apriltag.height, height_);
  CHECK(apriltag.height % 4 == 0);
  CHECK(apriltag.width % 4 == 0);

  CHECK_GE(apriltag.stride, width_);
  CHECK(apriltag.data != nullptr);
  ImageView gpu_image = apriltag;
  if (gpu_image.data_gpu == nullptr) {
    if (apriltag.stride == width_) {
      std::memcpy(input_view_.data, apriltag.data, static_cast<size_t>(width_) * height_);
    } else {
      for (int row = 0; row < height_; ++row) {
        std::memcpy(input_view_.data + static_cast<size_t>(row) * width_,
                    apriltag.data + static_cast<size_t>(row) * apriltag.stride, width_);
      }
    }
    gpu_image = input_view_;
  }
  PopulatePreprocessedApriltagGPU(gpu_image, min_view_, max_view_,
                                   threshold_view_, valid_view_, device_image_);
  PopulateSegmentedApriltagDevice(device_image_, device_labels_, width_, height_);
  has_frame_ = true;
  segmented_host_dirty_ = true;

  auto segments = segment_extractor_.ExtractDevice(
      device_labels_, width_, height_, width_);
  if (generate_debug_image) {
    GetSegmentedApriltagView();
    CUDA_CHECK(cudaMemcpy(binarized_apriltag_view_.data, device_image_,
                          static_cast<size_t>(width_) * height_, cudaMemcpyDeviceToHost));
    std::memset(boundary_segmented_apriltag_view_.data, 0,boundary_segmented_apriltag_view_.height * boundary_segmented_apriltag_view_.width * sizeof(uint32_t));
    PopulateBoundarySegmentedApriltag(segments,
                                      boundary_segmented_apriltag_view_);
  }

  segment_sorter_.Sort(segments);

  if (generate_debug_image) {
    std::memset(sorted_boundary_segmented_apriltag_view_.data, 0,sorted_boundary_segmented_apriltag_view_.height * sorted_boundary_segmented_apriltag_view_.width * sizeof(uint8_t));
    PopulateSortedBoundarySegmentedApriltag(
        segments, sorted_boundary_segmented_apriltag_view_);
  }

  auto candidate_quad_corners = GetCandidatesQuadCornersParallel(segments);
  CHECK_EQ(candidate_quad_corners.size(), segments.size());

  auto quads = GetQuads(candidate_quad_corners);

  if (generate_debug_image) {
    memcpy(candidate_quad_corners_apriltag_view_.data,
           sorted_boundary_segmented_apriltag_view_.data,
           sizeof(uint8_t) * apriltag.width * apriltag.height);
    PopulateCandidateQuadCornersApriltagBuffer(candidate_quad_corners, candidate_quad_corners_apriltag_view_);

    memcpy(quad_apriltag_view_.data, sorted_boundary_segmented_apriltag_view_.data,
           sizeof(uint8_t) * apriltag.width * apriltag.height);
    PopulateQuadApriltagBuffer(quads, quad_apriltag_view_);
  }

  auto bit_locations = GetBitLocations(quads);

  if (generate_debug_image) {
    memcpy(bit_locations_apriltag_view_.data, boundary_segmented_apriltag_view_.data,
           sizeof(uint32_t) * apriltag.width * apriltag.height);
    PopulateBitLocationsApriltag(bit_locations, bit_locations_apriltag_view_);
  }

  auto [tag_ids, rotations] = tag_decoder_.Decode(bit_locations, gpu_image);

  RotateQuads(quads, rotations);

  std::vector<ApriltagDetection> detections;
  CHECK_EQ(tag_ids.size(), rotations.size());
  for (size_t i = 0; i < tag_ids.size(); i++) {
    if (tag_ids[i] != -1) {
      detections.emplace_back(quads[i], tag_ids[i]);
    }
  }

  auto refined_points = GetRefinedPoints(detections, apriltag);

  if (generate_debug_image) {
    memcpy(refined_points_apriltag_view_.data,
           sorted_boundary_segmented_apriltag_view_.data,
           sizeof(uint8_t) * apriltag.width * apriltag.height);

    PopulateRefinedPointsApriltag(refined_points, refined_points_apriltag_view_);
  }

  auto refined_quads = GetRefinedQuads(refined_points);

  std::vector<ApriltagDetection> refined_detections;
  size_t refined_index = 0;
  for (int& i : tag_ids) {
    if (i != -1) {
      refined_detections.emplace_back(refined_quads[refined_index], i);
      ++refined_index;
    }
  }

  return refined_detections;
}

auto GPUApriltagDetector::GetSegmentedApriltagView() const -> ImageView32 {
  if (segmented_host_dirty_) {
    CUDA_CHECK(cudaMemcpy(segmented_apriltag_view_.data, device_labels_,
        static_cast<size_t>(width_) * height_ * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    segmented_host_dirty_ = false;
  }
  return segmented_apriltag_view_;
}

void GPUApriltagDetector::WriteLogImages(const std::filesystem::path& log_path) const {
  GetSegmentedApriltagView();
  if (has_frame_) {
    CUDA_CHECK(cudaMemcpy(binarized_apriltag_view_.data, device_image_,
                          static_cast<size_t>(width_) * height_, cudaMemcpyDeviceToHost));
  }
  ImWrite(log_path / "max.png", max_view_);
  ImWrite(log_path / "min.png", min_view_);
  ImWrite(log_path / "threshold.png", threshold_view_);
  ImWrite(log_path / "valid.png", valid_view_);
  ImWrite(log_path / "binarized_apriltag.png", binarized_apriltag_view_);
  ImWrite(log_path / "segmented_apriltag.png", segmented_apriltag_view_);
  ImWrite(log_path / "boundary_segmented_apriltag.png",
        boundary_segmented_apriltag_view_);
  ImWrite(log_path / "sorted_boundary_segmented_apriltag.png",
        sorted_boundary_segmented_apriltag_view_);
  ImWrite(log_path / "candidate_quad_corners_apriltag.png",
        candidate_quad_corners_apriltag_view_);
  ImWrite(log_path / "quad_apriltag.png", quad_apriltag_view_);
  ImWrite(log_path / "bit_locations_apriltag.png", bit_locations_apriltag_view_);
  ImWrite(log_path / "refined_points_apriltag.png", refined_points_apriltag_view_);
}

}  // namespace apriltag
