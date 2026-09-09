#include "apriltag/gpu_apriltag_detector.h"
#include "control_loop/timer.h"

#include <tag36h11.h>
#include <cstring>
#include <chrono>
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

namespace {

int ValidatedReducedWidth(int width, int height, int decimate) {
  // Validate before division and before any detector member allocates CUDA memory.
  CHECK_GT(decimate, 0);
  CHECK_GT(width, 0);
  CHECK_GT(height, 0);
  CHECK_EQ(width % (4LL * decimate), 0);
  CHECK_EQ(height % (4LL * decimate), 0);
  CHECK_LE(static_cast<size_t>(width) * height,
           static_cast<size_t>(std::numeric_limits<int>::max() - 255));
  return width / decimate;
}

__global__ void Decimate2xKernel(const uint8_t* __restrict__ input, int in_stride,
                                 uint8_t* __restrict__ output, int out_width, int out_height) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= out_height || c >= out_width) return;
  const int in_r = r * 2;
  const int in_c = c * 2;
  const uint32_t p00 = input[in_r * in_stride + in_c];
  const uint32_t p01 = input[in_r * in_stride + in_c + 1];
  const uint32_t p10 = input[(in_r + 1) * in_stride + in_c];
  const uint32_t p11 = input[(in_r + 1) * in_stride + in_c + 1];
  output[r * out_width + c] = static_cast<uint8_t>((p00 + p01 + p10 + p11 + 2) >> 2);
}

__global__ void DecimateKernel(const uint8_t* __restrict__ input, int in_stride,
                               uint8_t* __restrict__ output, int out_width, int out_height,
                               int factor) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  const int r = blockIdx.y * blockDim.y + threadIdx.y;
  if (r >= out_height || c >= out_width) return;
  uint64_t sum = 0;
  for (int dr = 0; dr < factor; ++dr) {
    const int in_r = r * factor + dr;
    for (int dc = 0; dc < factor; ++dc) {
      sum += input[in_r * in_stride + (c * factor + dc)];
    }
  }
  const uint64_t area = uint64_t(factor) * factor;
  output[r * out_width + c] = static_cast<uint8_t>((sum + area / 2) / area);
}

}  // namespace

namespace apriltag {
GPUApriltagDetector::GPUApriltagDetector(int width, int height,
                                         std::vector<int> target_tag_ids,
                                         int decimate)
    : width_(ValidatedReducedWidth(width, height, decimate)),
      height_(height / decimate),
      full_width_(width),
      full_height_(height),
      decimate_(decimate),
      target_tag_ids_(std::move(target_tag_ids)),
      family_(tag36h11_create()) {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));
  const int stride = width_;
  uint8_t* input_buffer = nullptr;
  CUDA_CHECK(cudaHostAlloc(&input_buffer, static_cast<size_t>(full_width_) * full_height_,
                           cudaHostAllocMapped));
  input_view_ = ImageView{.data = input_buffer, .stride = full_width_,
                          .height = full_height_, .width = full_width_};
  input_view_.EnableGpu();

  const int tile_w = width_ / 4;
  const int tile_h = height_ / 4;

  uint8_t* max_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&max_buffer),
                           static_cast<size_t>(tile_w) * tile_h, cudaHostAllocMapped));
  uint8_t* min_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&min_buffer),
                           static_cast<size_t>(tile_w) * tile_h, cudaHostAllocMapped));
  max_view_ = ImageView{.data = max_buffer,
                        .stride = tile_w,
                        .height = tile_h,
                        .width = tile_w};
  min_view_ = ImageView{.data = min_buffer,
                        .stride = tile_w,
                        .height = tile_h,
                        .width = tile_w};
  max_view_.EnableGpu();
  min_view_.EnableGpu();

  uint8_t* threshold_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&threshold_buffer),
                           static_cast<size_t>(tile_w) * tile_h, cudaHostAllocMapped));
  threshold_view_ = ImageView{.data = threshold_buffer,
                              .stride = tile_w,
                              .height = tile_h,
                              .width = tile_w};
  uint8_t* valid_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&valid_buffer),
                           static_cast<size_t>(tile_w) * tile_h, cudaHostAllocMapped));
  valid_view_ = ImageView{.data = valid_buffer,
                          .stride = tile_w,
                          .height = tile_h,
                          .width = tile_w};
  threshold_view_.EnableGpu();
  valid_view_.EnableGpu();

  auto* binarized_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width_ * height_, sizeof(uint8_t)));
  binarized_apriltag_view_ = ImageView{.data = binarized_apriltag_buffer,
                                       .stride = stride,
                                       .height = height_,
                                       .width = width_};

  auto* segmented_apriltag_buffer =
      static_cast<uint32_t*>(calloc(width_ * height_, sizeof(uint32_t)));
  segmented_apriltag_view_ = ImageView32{.data = segmented_apriltag_buffer,
                                         .stride = stride,
                                         .height = height_,
                                         .width = width_};

  auto* boundary_segmented_apriltag_buffer =
      static_cast<uint32_t*>(calloc(width_ * height_, sizeof(uint32_t)));
  boundary_segmented_apriltag_view_ =
      ImageView32{.data = boundary_segmented_apriltag_buffer,
                  .stride = stride,
                  .height = height_,
                  .width = width_};

  auto* sorted_boundary_segmented_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width_ * height_, sizeof(uint8_t)));
  sorted_boundary_segmented_apriltag_view_ =
      ImageView{.data = sorted_boundary_segmented_apriltag_buffer,
                .stride = stride,
                .height = height_,
                .width = width_};

  auto* candidate_quad_corners_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width_ * height_, sizeof(uint8_t)));
  candidate_quad_corners_apriltag_view_ =
      ImageView{.data = candidate_quad_corners_apriltag_buffer,
                .stride = stride,
                .height = height_,
                .width = width_};

  auto* quad_apriltag_buffer =
      static_cast<uint8_t*>(calloc(width_ * height_, sizeof(uint8_t)));
  quad_apriltag_view_ = ImageView{.data = quad_apriltag_buffer,
                                  .stride = stride,
                                  .height = height_,
                                  .width = width_};

  auto* bit_locations_apriltag_buffer =
      static_cast<uint32_t*>(calloc(full_width_ * full_height_, sizeof(uint32_t)));
  bit_locations_apriltag_view_ = ImageView32{
      .data = bit_locations_apriltag_buffer,
      .stride = full_width_,
      .height = full_height_,
      .width = full_width_,
  };

  auto* refined_points_apriltag_buffer =
      static_cast<uint8_t*>(calloc(full_width_ * full_height_, sizeof(uint8_t)));
  refined_points_apriltag_view_ =
      ImageView{.data = refined_points_apriltag_buffer,
                .stride = full_width_,
                .height = full_height_,
                .width = full_width_};

  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_image_),
                        width_ * height_ * sizeof(uint8_t)));
  if (decimate_ > 1) {
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_reduced_image_),
                          width_ * height_ * sizeof(uint8_t)));
  }
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_labels_),
                        width_ * height_ * sizeof(uint32_t)));
  tag_decoder_.SetTargetCodes(family_, target_tag_ids_);
}
GPUApriltagDetector::~GPUApriltagDetector() {
  cudaFreeHost(input_view_.data);
  cudaFree(device_image_);
  if (device_reduced_image_) cudaFree(device_reduced_image_);
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
auto GPUApriltagDetector::Detect(ImageView apriltag, bool generate_debug_image,
                                  Profile* profile)
    -> std::vector<ApriltagDetection> {
  auto stage_start = profile ? std::chrono::steady_clock::now()
                             : std::chrono::steady_clock::time_point{};
  auto stage = [&](double Profile::*field) {
    if (!profile) return;
    const auto now = std::chrono::steady_clock::now();
    profile->*field = std::chrono::duration<double, std::milli>(now - stage_start).count();
    stage_start = now;
  };
  CHECK_EQ(apriltag.width, full_width_);
  CHECK_EQ(apriltag.height, full_height_);

  CHECK_GE(apriltag.stride, full_width_);
  CHECK(apriltag.data != nullptr);
  ImageView gpu_image = apriltag;
  if (gpu_image.data_gpu == nullptr) {
    if (apriltag.stride == full_width_) {
      std::memcpy(input_view_.data, apriltag.data, static_cast<size_t>(full_width_) * full_height_);
    } else {
      for (int row = 0; row < full_height_; ++row) {
        std::memcpy(input_view_.data + static_cast<size_t>(row) * full_width_,
                    apriltag.data + static_cast<size_t>(row) * apriltag.stride, full_width_);
      }
    }
    gpu_image = input_view_;
  }

  ImageView reduced_gpu_image = gpu_image;
  if (decimate_ > 1) {
    constexpr dim3 block(32, 8);
    const dim3 grid((width_ + block.x - 1) / block.x,
                    (height_ + block.y - 1) / block.y);
    if (decimate_ == 2) {
      Decimate2xKernel<<<grid, block>>>(gpu_image.data_gpu, gpu_image.stride,
                                        device_reduced_image_, width_, height_);
    } else {
      DecimateKernel<<<grid, block>>>(gpu_image.data_gpu, gpu_image.stride,
                                      device_reduced_image_, width_, height_, decimate_);
    }
    CUDA_CHECK(cudaGetLastError());
    reduced_gpu_image = ImageView{
        .data = nullptr,
        .data_gpu = device_reduced_image_,
        .stride = width_,
        .height = height_,
        .width = width_,
    };
  }

  PopulatePreprocessedApriltagGPU(reduced_gpu_image, min_view_, max_view_,
                                 threshold_view_, valid_view_, device_image_);
  PopulateSegmentedApriltagDevice(device_image_, device_labels_, width_, height_);
  has_frame_ = true;
  segmented_host_dirty_ = true;

  const int min_boundary_count = std::max(16, 40 / decimate_);
  auto segments = segment_extractor_.ExtractDevice(
      device_labels_, width_, height_, width_, min_boundary_count);
  if (generate_debug_image) {
    GetSegmentedApriltagView();
    CUDA_CHECK(cudaMemcpy(binarized_apriltag_view_.data, device_image_,
                          static_cast<size_t>(width_) * height_, cudaMemcpyDeviceToHost));
    std::memset(boundary_segmented_apriltag_view_.data, 0,
                boundary_segmented_apriltag_view_.height * boundary_segmented_apriltag_view_.width * sizeof(uint32_t));
    PopulateBoundarySegmentedApriltag(segments,
                                      boundary_segmented_apriltag_view_);
  }

  stage(&Profile::extract_ms);
  segment_sorter_.Sort(segments);
  stage(&Profile::sort_ms);

  if (generate_debug_image) {
    std::memset(sorted_boundary_segmented_apriltag_view_.data, 0,
                sorted_boundary_segmented_apriltag_view_.height * sorted_boundary_segmented_apriltag_view_.width * sizeof(uint8_t));
    PopulateSortedBoundarySegmentedApriltag(
        segments, sorted_boundary_segmented_apriltag_view_);
  }

  auto candidate_quad_corners = GetCandidatesQuadCornersParallel(segments);
  CHECK_EQ(candidate_quad_corners.size(), segments.size());

  auto quads = GetQuads(candidate_quad_corners);

  if (generate_debug_image) {
    memcpy(candidate_quad_corners_apriltag_view_.data,
           sorted_boundary_segmented_apriltag_view_.data,
           sizeof(uint8_t) * width_ * height_);
    PopulateCandidateQuadCornersApriltagBuffer(candidate_quad_corners, candidate_quad_corners_apriltag_view_);

    memcpy(quad_apriltag_view_.data, sorted_boundary_segmented_apriltag_view_.data,
           sizeof(uint8_t) * width_ * height_);
    PopulateQuadApriltagBuffer(quads, quad_apriltag_view_);
  }

  if (decimate_ > 1) {
    for (auto& quad : quads) {
      for (auto& point : quad.corners) {
        point.row *= decimate_;
        point.col *= decimate_;
      }
    }
  }

  auto bit_locations = GetBitLocations(quads);

  if (generate_debug_image) {
    std::memset(bit_locations_apriltag_view_.data, 0,
                sizeof(uint32_t) * full_width_ * full_height_);
    PopulateBitLocationsApriltag(bit_locations, bit_locations_apriltag_view_);
  }

  stage(&Profile::quads_ms);
  std::vector<int> hammings;
  auto [tag_ids, rotations] = tag_decoder_.Decode(bit_locations, gpu_image,
                                                  decimate_ > 1 ? &hammings : nullptr);

  // For near-miss quads that failed to decode under decimation (where coarse corners
  // have quantization error), try refining their corners on the full-resolution image and re-decoding.
  if (decimate_ > 1) {
    std::vector<int> retry_indices;
    std::vector<Quad> retry_quads;
    for (size_t i = 0; i < tag_ids.size(); ++i) {
      if (tag_ids[i] == -1 && hammings[i] <= 6) {
        retry_indices.push_back(static_cast<int>(i));
        retry_quads.push_back(quads[i]);
      }
    }

    if (!retry_quads.empty()) {
      auto retry_refined_points = GetRefinedPoints(retry_quads, apriltag);
      auto retry_refined_quads = GetRefinedQuads(retry_refined_points, retry_quads);
      auto retry_bit_locations = GetBitLocations(retry_refined_quads);
      auto [retry_ids, retry_rots] = tag_decoder_.Decode(retry_bit_locations, gpu_image);
      for (size_t r = 0; r < retry_indices.size(); ++r) {
        if (retry_ids[r] != -1) {
          const int idx = retry_indices[r];
          tag_ids[idx] = retry_ids[r];
          rotations[idx] = retry_rots[r];
          quads[idx] = retry_refined_quads[r];
        }
      }
    }
  }

  stage(&Profile::decode_ms);
  std::vector<Quad> decoded_quads;
  std::vector<int> decoded_ids;
  std::vector<int> decoded_rotations;
  for (size_t i = 0; i < tag_ids.size(); ++i) {
    if (tag_ids[i] != -1) {
      decoded_quads.push_back(quads[i]);
      decoded_ids.push_back(tag_ids[i]);
      decoded_rotations.push_back(rotations[i]);
    }
  }

  auto final_refined_points = GetRefinedPoints(decoded_quads, apriltag);
  if (generate_debug_image) {
    std::memset(refined_points_apriltag_view_.data, 0,
                sizeof(uint8_t) * full_width_ * full_height_);
    PopulateRefinedPointsApriltag(final_refined_points, refined_points_apriltag_view_);
  }

  auto final_refined_quads = GetRefinedQuads(final_refined_points, decoded_quads);
  RotateQuads(final_refined_quads, decoded_rotations);

  std::vector<ApriltagDetection> detections;
  detections.reserve(decoded_ids.size());
  for (size_t i = 0; i < decoded_ids.size(); ++i) {
    detections.emplace_back(final_refined_quads[i], decoded_ids[i]);
  }

  stage(&Profile::refine_ms);
  return detections;
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
