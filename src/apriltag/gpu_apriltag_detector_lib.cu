#include "apriltag/gpu_apriltag_detector_lib.h"

#include <cstdio>
#include <cuda_runtime.h>
#include <cuda/std/algorithm>
#include <cuda/cmath>

#define CUDA_CHECK(expr_to_check)                                      \
    do {                                                               \
        const cudaError_t result = (expr_to_check);                    \
        if (result != cudaSuccess) {                                   \
            std::fprintf(                                              \
                stderr,                                                \
                "CUDA Runtime Error: %s:%d: %d = %s\n",                \
                __FILE__,                                              \
                __LINE__,                                              \
                static_cast<int>(result),                              \
                cudaGetErrorString(result));                           \
        }                                                              \
    } while (false)

#define CHECK(condition)                                               \
    do {                                                               \
        if (!(condition)) {                                            \
            std::fprintf(                                              \
                stderr,                                                \
                "Check failed: %s (%s:%d)\n",                          \
                #condition,                                            \
                __FILE__,                                              \
                __LINE__);                                             \
            std::abort();                                              \
        }                                                              \
    } while (false)

namespace {
  __host__ __device__ constexpr auto ceil_div(int n, int divisor) -> int {
      return n / divisor + (n % divisor != 0);
  }

  template <typename T>
  struct ImageViewGPU {
    ImageViewGPU(apriltag::ImageView<T> image_view) : data(image_view.data), stride(image_view.stride), height(image_view.height), width(image_view.width) {
      cudaPointerAttributes attributes{};
      CUDA_CHECK(cudaPointerGetAttributes(&attributes, image_view.data));
      if (attributes.type == cudaMemoryTypeHost){
        cudaHostGetDevicePointer(
          reinterpret_cast<void**>(&data),
          image_view.data,
          0
      );
      }
    }
    T* data;
    int stride;
    int height;
    int width;

    __device__ auto operator()(size_t row, size_t col) -> T&{
      return data[row * stride + col];
    }
  };
}

namespace{
  __global__ void PopulateMinMaxKernal(ImageViewGPU<uint8_t> apriltag, ImageViewGPU<uint8_t> min_view, ImageViewGPU<uint8_t> max_view){
    int min_max_col_index = threadIdx.x + blockIdx.x * blockDim.x;
    int min_max_row_index = threadIdx.y + blockIdx.y * blockDim.y;

    if (min_max_col_index >= min_view.width || min_max_row_index >= min_view.height){
      return;
    }

    int apriltag_row_index = min_max_row_index * 4;
    int apriltag_col_index = min_max_col_index * 4;

    uint8_t min_value = 255;
    uint8_t max_value = 0;

    for (int i = 0; i < 4; i++) {
      for (int j = 0; j < 4; j++) {
        auto value = apriltag(apriltag_row_index + i, apriltag_col_index + j);
        min_value = cuda::std::min(min_value, value);
        max_value = cuda::std::max(max_value, value);
      }
    }
    min_view(min_max_row_index, min_max_col_index) = min_value;
    max_view(min_max_row_index, min_max_col_index) = max_value;
  }

  __global__ void PopulateBinarizedApriltagKernal(ImageViewGPU<uint8_t> apriltag, ImageViewGPU<uint8_t> min_view, ImageViewGPU<uint8_t> max_view, ImageViewGPU<uint8_t> binarized_apriltag){
    int col_offset = threadIdx.x + blockIdx.x * blockDim.x;
    int row_offset = threadIdx.y + blockIdx.y * blockDim.y;

    if (row_offset >= min_view.height || col_offset >= min_view.width) {
      return;
    }

    uint8_t min_value = 255;
    uint8_t max_value = 0;
    for (int row = cuda::std::max(0, row_offset - 1); row <= cuda::std::min(min_view.height - 1, row_offset + 1); row++){
      for (int col = cuda::std::max(0, col_offset - 1); col <= cuda::std::min(min_view.width - 1, col_offset + 1); col++){
        min_value = cuda::std::min(min_value, min_view(row, col));
        max_value = cuda::std::max(max_value, max_view(row, col));
      }
    }
    uint8_t threshold = (max_value / 2) + (min_value / 2);
    uint8_t valid = max_value - min_value > 10 ? 255 : 0;

    if (valid == 0){
      for (int row = row_offset * 4; row < (row_offset * 4) + 4; row++){
        for (int col = col_offset * 4; col < (col_offset * 4) + 4; col++){
          binarized_apriltag(row, col) = apriltag(row, col) > threshold ? (255 / 2) + 50 : (255 / 2 - 50);
        }
      }
    } else{
      for (int row = row_offset * 4; row < (row_offset * 4) + 4; row++){
        for (int col = col_offset * 4; col < (col_offset * 4) + 4; col++){
          binarized_apriltag(row, col) = apriltag(row, col) > threshold ? 255 : 0;
        }
      }
    }
    return;
  }
  __global__ void InitDSUValidKernel(ImageViewGPU<uint8_t> binarized_apriltag, ImageViewGPU<uint32_t> dsu){
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= dsu.height || col >= dsu.width){
      return;
    }

    uint8_t value = binarized_apriltag(row, col);
    if (value != 0){
      // Invalid
      dsu(row, col) = UINT32_MAX;
      return;
    }

    bool valid = false;
    for (int i = cuda::std::max(0, static_cast<int>(row) - 1); i <= cuda::std::min(static_cast<int>(dsu.height - 1), static_cast<int>(row) + 1); i++){
      for (int j = cuda::std::max(0, static_cast<int>(col) - 1); j <= cuda::std::min(static_cast<int>(dsu.width - 1), static_cast<int>(col) + 1); j++){
        if (binarized_apriltag(i, j) == 255){
          valid = true;
          break;
        }
      }
    }
    if (!valid){
      // Invalid
      dsu(row, col) = UINT32_MAX;
      return;
    }
    dsu(row, col) = 0; // Valid
  }
  __global__ void InitDSUKernel(ImageViewGPU<uint8_t> binarized_apriltag, ImageViewGPU<uint32_t> dsu){
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;
    if (row >= dsu.height || col >= dsu.width){
      return;
    }

    if (dsu(row, col) == UINT32_MAX){
      return;
    }

    uint8_t value = binarized_apriltag(row, col);
    int stride = dsu.stride;
    if (row + 1 < binarized_apriltag.height && binarized_apriltag(row + 1, col) == value && dsu(row + 1, col) != UINT32_MAX){
      bool valid_join = false;
      if (col > 0){
        if (binarized_apriltag(row, col - 1) == 255 || binarized_apriltag(row + 1, col - 1) == 255){
          valid_join = true;
        }
      }
      if (col + 1 < binarized_apriltag.width){
        if (binarized_apriltag(row, col + 1) == 255 || binarized_apriltag(row + 1, col + 1) == 255){
          valid_join = true;
        }
      }
      if (valid_join){
        dsu(row, col) = (row + 1) * stride + col; 
        return;
      }
    }
    if (col + 1 < binarized_apriltag.width && binarized_apriltag(row, col + 1) == value && dsu(row, col + 1) != UINT32_MAX){
      bool valid_join = false;
      if (row > 0){
        if (binarized_apriltag(row - 1, col) == 255 || binarized_apriltag(row - 1, col + 1) == 255){
          valid_join = true;
        }
      }
      if (row + 1 < binarized_apriltag.height){
        if (binarized_apriltag(row + 1, col) == 255 || binarized_apriltag(row + 1, col + 1) == 255){
          valid_join = true;
        }
      }
      if (valid_join){
        dsu(row, col) = row * stride + col + 1; 
        return;
      }
    }

    dsu(row, col) = row * stride + col; 
  }
  __global__ void FlattenDSUKernel(ImageViewGPU<uint32_t> dsu){
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row >= dsu.height || col >= dsu.width){
      return;
    }

    uint32_t dsu_value = dsu(row, col);
    if (dsu_value == UINT32_MAX){
      // Invalid
      return;
    }
    if (dsu_value != row * dsu.stride + col){
      dsu(row, col) = dsu.data[dsu_value];
    }
  }

  __device__ auto GetRoot(uint32_t curr_row, uint32_t curr_col, ImageViewGPU<uint32_t> dsu) -> uint32_t{
      while (true){
        uint32_t dsu_value = dsu(curr_row, curr_col);
        uint32_t next_row = dsu_value / dsu.stride;
        uint32_t next_col = dsu_value % dsu.stride;
        if (next_row == curr_row && next_col == curr_col){
          break;
        }
        curr_row = next_row;
        curr_col = next_col;
      }
      return curr_row * dsu.stride + curr_col;
  }

  __global__ void JoinDSUKernel(ImageViewGPU<uint8_t> binarized_apriltag, ImageViewGPU<uint32_t> dsu){
    uint32_t row = threadIdx.y + blockIdx.y * blockDim.y;
    uint32_t col = threadIdx.x + blockIdx.x * blockDim.x;

    if (row + 1 >= dsu.height || col + 1 >= dsu.width){
      return;
    }

    if (dsu(row, col) == UINT32_MAX){
      // Invalid
      return;
    }

    uint8_t value = binarized_apriltag(row, col);
    // if (0 != dsu(row + 1, col) && 0 != dsu(row, col + 1)){
    // if (value == binarized_apriltag(row + 1, col) && value == binarized_apriltag(row, col + 1)){
    if (UINT32_MAX != dsu(row + 1, col) && UINT32_MAX != dsu(row, col + 1)){
      bool valid_join = false;
      if (row > 0){
        if (binarized_apriltag(row - 1, col) == 255 || binarized_apriltag(row - 1, col + 1) == 255){
          valid_join = true;
        }
      }
      if (row + 1 < binarized_apriltag.height){
        if (binarized_apriltag(row + 1, col) == 255 || binarized_apriltag(row + 1, col + 1) == 255){
          valid_join = true;
        }
      }
      if (valid_join){
        while(true){
          uint32_t larger_index = GetRoot(row, col + 1, dsu);
          uint32_t smaller_index = GetRoot(row + 1, col, dsu);
          if (larger_index == smaller_index){
            return;
          }
          if (larger_index < smaller_index){
            cuda::std::swap(larger_index, smaller_index);
          }
          if(atomicCAS(dsu.data + smaller_index, smaller_index, larger_index) == smaller_index){
            // Set succesfully
            break;
          }
        }
      }
    }
  }
}

namespace apriltag{
  void GpuApriltagDetector::PopulateMinMaxGPU(ImageView<uint8_t> apriltag, ImageView<uint8_t> min, ImageView<uint8_t> max, cudaStream_t stream){
    ImageViewGPU<uint8_t> apriltag_gpu(apriltag);
    ImageViewGPU<uint8_t> min_gpu(min);
    ImageViewGPU<uint8_t> max_gpu(max);

    dim3 threads(32, 8);
    dim3 blocks(ceil_div(min.width, threads.x), ceil_div(min.height, threads.y));
    PopulateMinMaxKernal<<<blocks, threads, 0, stream>>>(apriltag_gpu, min_gpu, max_gpu);
  }

  void GpuApriltagDetector::PopulateThresholdValidGPU(ImageView<uint8_t> apriltag, ImageView<uint8_t> min, ImageView<uint8_t> max, ImageView<uint8_t> binarized_apriltag,
                                 cudaStream_t stream){
    ImageViewGPU<uint8_t> d_apriltag(apriltag);
    ImageViewGPU<uint8_t> d_min(min);
    ImageViewGPU<uint8_t> d_max(max);
    ImageViewGPU<uint8_t> d_binarized_apriltag(binarized_apriltag);
    dim3 threads(8, 8);
    dim3 blocks(ceil_div(min.width, threads.x), ceil_div(min.height, threads.y));
    PopulateBinarizedApriltagKernal<<<blocks, threads, 0, stream>>>(d_apriltag, d_min, d_max, d_binarized_apriltag);
  }

  void GpuApriltagDetector::PopulateSegmentedApriltagGPU(ImageView<uint8_t> binarized_apriltag,
                                    ImageView<uint32_t> segmented_apriltag, ImageView<uint32_t> dsu, cudaStream_t stream){
    ImageViewGPU<uint8_t> d_binarized_apriltag(binarized_apriltag);
    ImageViewGPU<uint32_t> d_segmented_apriltag(segmented_apriltag);
    ImageViewGPU<uint32_t> d_dsu(dsu);
    dim3 threads(4, 32);
    dim3 blocks(ceil_div(dsu.width, threads.x), ceil_div(dsu.height, threads.y));
    InitDSUValidKernel<<<blocks, threads, 0, stream>>>(d_binarized_apriltag, d_dsu);
    InitDSUKernel<<<blocks, threads, 0, stream>>>(d_binarized_apriltag, d_dsu);

    for (int i = 0; i < 8; i++){
      FlattenDSUKernel<<<blocks, threads, 0, stream>>>(d_dsu);
    }
    JoinDSUKernel<<<blocks, threads, 0, stream>>>(binarized_apriltag, dsu);
    for (int i = 0; i < 8; i++){
      FlattenDSUKernel<<<blocks, threads, 0, stream>>>(d_dsu);
    }
    return;
  }


  void GpuApriltagDetector::RegisterApriltagViewToGPU(ImageView<uint8_t> apriltag){
    CUDA_CHECK(cudaHostRegister(
          apriltag.data, apriltag.stride * apriltag.height * sizeof(uint8_t),
          cudaHostRegisterMapped));
  }


  void GpuApriltagDetector::UnregisterApriltagViewToGPU(ImageView<uint8_t> apriltag){
    cudaHostUnregister(apriltag.data);
  }

  void GpuApriltagDetector::SyncStream(){
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

}
