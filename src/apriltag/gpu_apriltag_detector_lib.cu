#include "apriltag/gpu_apriltag_detector_lib.h"

#include "absl/log/check.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>

#include <cuda/std/algorithm>

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    const cudaError_t cuda_check_error = (call);                           \
    if (cuda_check_error != cudaSuccess) {                                 \
      std::cerr << cudaGetErrorString(cuda_check_error) << '\n';            \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

namespace {

constexpr uint32_t kInvalidLabel = std::numeric_limits<uint32_t>::max();

__device__ auto Get(uint8_t* data_gpu, int stride, size_t row, size_t col)
    -> uint8_t& {
  return data_gpu[row * stride + col];
}

__global__ void MinMaxKernel(apriltag::ImageView apriltag,
                             apriltag::ImageView min_image_view,
                             apriltag::ImageView max_image_view, int rows,
                             int cols) {
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= rows || col >= cols) {
    return;
  }

  const uint apriltag_row = row * 4;
  const uint apriltag_col = col * 4;

  uint8_t min = 255;
  uint8_t max = 0;
#pragma unroll
  for (int i = 0; i < 4; i++) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
      const uint8_t pixel = Get(apriltag.data_gpu, apriltag.stride,
                                apriltag_row + i, apriltag_col + j);
      min = cuda::std::min(min, pixel);
      max = cuda::std::max(max, pixel);
    }
  }
  Get(min_image_view.data_gpu, min_image_view.stride, row, col) = min;
  Get(max_image_view.data_gpu, max_image_view.stride, row, col) = max;
}

__global__ void ThresholdValidKernel(apriltag::ImageView min_image_view,
                                     apriltag::ImageView max_image_view,
                                     apriltag::ImageView threshold_image_view,
                                     apriltag::ImageView valid_image_view) {
  const uint col = blockIdx.x * blockDim.x + threadIdx.x;
  const uint row = blockIdx.y * blockDim.y + threadIdx.y;

  if (row >= min_image_view.height || col >= min_image_view.width) {
    return;
  }

  uint8_t& threshold = Get(threshold_image_view.data_gpu,
                           threshold_image_view.stride, row, col);
  uint8_t& valid =
      Get(valid_image_view.data_gpu, valid_image_view.stride, row, col);
  if (row == 0 || col == 0 || row == min_image_view.height - 1 ||
      col == min_image_view.width - 1) {
    const uint8_t min =
        Get(min_image_view.data_gpu, min_image_view.stride, row, col);
    const uint8_t max =
        Get(max_image_view.data_gpu, max_image_view.stride, row, col);
    threshold = (min / 2) + (max / 2);
    valid = 0;
    return;
  }

  uint8_t min = 255;
  uint8_t max = 0;
#pragma unroll
  for (int row_offset = -1; row_offset <= 1; ++row_offset) {
#pragma unroll
    for (int col_offset = -1; col_offset <= 1; ++col_offset) {
      min = cuda::std::min(
          min, Get(min_image_view.data_gpu, min_image_view.stride,
                   row + row_offset, col + col_offset));
      max = cuda::std::max(
          max, Get(max_image_view.data_gpu, max_image_view.stride,
                   row + row_offset, col + col_offset));
    }
  }
  threshold = (max / 2) + (min / 2);
  valid = max - min > 25 ? 255 : 0;
}

__device__ __forceinline__ auto FindRoot(uint32_t* labels, uint32_t label)
    -> uint32_t {
  uint32_t next = labels[label];
  while (label != next) {
    label = next;
    next = labels[label];
  }
  return label;
}

// Merge two label trees by always attaching the larger root to the smaller
// root. Another thread may update either tree while it is being traversed, so
// the root update must be atomic.
__device__ __forceinline__ auto ReduceLabels(uint32_t* labels,
                                             uint32_t label_1,
                                             uint32_t label_2) -> uint32_t {
  uint32_t next_1 = label_1 != label_2 ? labels[label_1] : 0;
  uint32_t next_2 = label_1 != label_2 ? labels[label_2] : 0;

  while (label_1 != label_2 && label_1 != next_1) {
    label_1 = next_1;
    next_1 = labels[label_1];
  }
  while (label_1 != label_2 && label_2 != next_2) {
    label_2 = next_2;
    next_2 = labels[label_2];
  }

  while (label_1 != label_2) {
    if (label_1 < label_2) {
      const uint32_t temporary = label_1;
      label_1 = label_2;
      label_2 = temporary;
    }

    const uint32_t previous = atomicMin(&labels[label_1], label_2);
    label_1 = label_1 == previous ? label_2 : previous;
  }
  return label_1;
}

__global__ void InitializeLabelsKernel(const uint8_t* image, uint32_t* labels,
                                       int width, int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= height || col >= width) {
    return;
  }

  const uint32_t index = row * width + col;
  const uint8_t pixel = image[index];
  if (pixel != 0 && pixel != 255) {
    labels[index] = kInvalidLabel;
    return;
  }

  const bool connected_left = col > 0 && pixel == image[index - 1];
  const bool connected_up = row > 0 && pixel == image[index - width];

  uint32_t label = connected_left ? index - 1 : index;
  // The upper pixel always has a smaller linear index than the left pixel.
  label = connected_up ? index - width : label;
  labels[index] = label;
}

__global__ void ResolveLabelsKernel(uint32_t* labels, int pixel_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count || labels[index] == kInvalidLabel) {
    return;
  }
  labels[index] = FindRoot(labels, labels[index]);
}

__global__ void ReduceCriticalLabelsKernel(const uint8_t* image,
                                           uint32_t* labels, int width,
                                           int height) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row <= 0 || row >= height || col <= 0 || col >= width) {
    return;
  }

  const uint32_t index = row * width + col;
  const uint8_t pixel = image[index];
  if (pixel != 0 && pixel != 255) {
    return;
  }

  const bool connected_left = pixel == image[index - 1];
  const bool connected_up = pixel == image[index - width];
  const bool connected_upper_left = pixel == image[index - width - 1];
  if (connected_left && connected_up && !connected_upper_left) {
    ReduceLabels(labels, labels[index], labels[index - 1]);
  }
}

__global__ void EncodeResolvedLabelsKernel(uint32_t* labels,
                                           uint32_t* output,
                                           int pixel_count) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= pixel_count) {
    return;
  }

  const uint32_t label = labels[index];
  // The CPU pipeline reserves zero for invalid/unlabeled pixels. Internally,
  // Playne labels are zero-based pixel indices, so encode roots as index + 1.
  output[index] = label == kInvalidLabel ? 0 : label + uint32_t{1};
}

}  // namespace

namespace apriltag {

void ImageView::EnableGpu() {
  CUDA_CHECK(cudaHostGetDevicePointer(reinterpret_cast<void**>(&data_gpu), data,
                                      0));
}

void PopulateMinMaxGPU(ImageView apriltag, ImageView min, ImageView max) {
  CHECK(apriltag.data_gpu != nullptr);
  CHECK(min.data_gpu != nullptr);
  CHECK(max.data_gpu != nullptr);
  constexpr dim3 block(32, 8);
  const dim3 grid((min.width + block.x - 1) / block.x,
                  (min.height + block.y - 1) / block.y);

  MinMaxKernel<<<grid, block, 0>>>(apriltag, min, max, min.height, min.width);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void PopulateThresholdValidGPU(ImageView min, ImageView max,
                               ImageView threshold, ImageView valid) {
  CHECK_EQ(min.width, threshold.width);
  CHECK_EQ(min.height, threshold.height);
  CHECK_EQ(max.width, threshold.width);
  CHECK_EQ(max.height, threshold.height);
  CHECK_EQ(valid.width, threshold.width);
  CHECK_EQ(valid.height, threshold.height);
  CHECK(min.data_gpu != nullptr);
  CHECK(max.data_gpu != nullptr);
  CHECK(threshold.data_gpu != nullptr);
  CHECK(valid.data_gpu != nullptr);

  constexpr dim3 block(32, 8);
  const dim3 grid((min.width + block.x - 1) / block.x,
                  (min.height + block.y - 1) / block.y);
  ThresholdValidKernel<<<grid, block>>>(min, max, threshold, valid);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void PopulateSegmentedApriltagGPU(ImageView binarized_apriltag,
                                  ImageView32 segmented_apriltag) {
  CHECK_EQ(binarized_apriltag.width, segmented_apriltag.width);
  CHECK_EQ(binarized_apriltag.height, segmented_apriltag.height);
  CHECK_GE(binarized_apriltag.stride, binarized_apriltag.width);
  CHECK_GE(segmented_apriltag.stride, segmented_apriltag.width);

  const int width = binarized_apriltag.width;
  const int height = binarized_apriltag.height;
  if (width == 0 || height == 0) {
    return;
  }

  const size_t pixel_count = static_cast<size_t>(width) * height;
  CHECK_LE(pixel_count,
           static_cast<size_t>(std::numeric_limits<int>::max()));
  uint8_t* device_image = nullptr;
  uint32_t* device_labels = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_image), pixel_count));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_labels),
                        pixel_count * sizeof(uint32_t)));

  CUDA_CHECK(cudaMemcpy2D(device_image, width, binarized_apriltag.data,
                          binarized_apriltag.stride, width, height,
                          cudaMemcpyHostToDevice));

  constexpr dim3 block(32, 8);
  const dim3 grid((width + block.x - 1) / block.x,
                  (height + block.y - 1) / block.y);
  InitializeLabelsKernel<<<grid, block>>>(device_image, device_labels, width,
                                          height);
  CUDA_CHECK(cudaGetLastError());

  constexpr int resolve_block_size = 256;
  const int resolve_grid_size =
      (static_cast<int>(pixel_count) + resolve_block_size - 1) /
      resolve_block_size;
  ResolveLabelsKernel<<<resolve_grid_size, resolve_block_size>>>(
      device_labels, static_cast<int>(pixel_count));
  CUDA_CHECK(cudaGetLastError());

  ReduceCriticalLabelsKernel<<<grid, block>>>(device_image, device_labels,
                                               width, height);
  CUDA_CHECK(cudaGetLastError());

  ResolveLabelsKernel<<<resolve_grid_size, resolve_block_size>>>(
      device_labels, static_cast<int>(pixel_count));
  CUDA_CHECK(cudaGetLastError());

  EncodeResolvedLabelsKernel<<<resolve_grid_size, resolve_block_size>>>(
      device_labels, device_labels, static_cast<int>(pixel_count));
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpy2D(segmented_apriltag.data,
                          static_cast<size_t>(segmented_apriltag.stride) *
                              sizeof(uint32_t),
                          device_labels, width * sizeof(uint32_t),
                          width * sizeof(uint32_t), height,
                          cudaMemcpyDeviceToHost));

  CUDA_CHECK(cudaFree(device_labels));
  CUDA_CHECK(cudaFree(device_image));
}

}  // namespace apriltag
