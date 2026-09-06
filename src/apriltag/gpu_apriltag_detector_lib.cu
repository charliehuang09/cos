#include "apriltag/gpu_apriltag_detector_lib.h"

#include "absl/log/check.h"

#include <cstdlib>
#include <iostream>

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

}  // namespace apriltag
