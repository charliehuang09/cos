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
  struct ImageViewGPU {
    ImageViewGPU(apriltag::ImageView image_view) : data(image_view.data), stride(image_view.stride), height(image_view.height), width(image_view.width) {
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
    uint8_t* data;
    int stride;
    int height;
    int width;

    __device__ auto operator()(size_t row, size_t col) -> uint8_t&{
      return data[row * stride + col];
    }
  };
}

namespace{
  __global__ void PopulateMinMaxKernal(ImageViewGPU apriltag, ImageViewGPU min_view, ImageViewGPU max_view){
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

  __global__ void PopulateBinarizedApriltagKernal(ImageViewGPU apriltag, ImageViewGPU min_view, ImageViewGPU max_view, ImageViewGPU binarized_apriltag){
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
    uint8_t valid = max_value - min_value > 50 ? 255 : 0;

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
}

namespace apriltag{
  void GpuApriltagDetector::PopulateMinMaxGPU(ImageView apriltag, ImageView min, ImageView max, cudaStream_t stream){
    ImageViewGPU apriltag_gpu(apriltag);
    ImageViewGPU min_gpu(min);
    ImageViewGPU max_gpu(max);

    dim3 threads(32, 8);
    dim3 blocks(cuda::ceil_div(min.width, threads.x), cuda::ceil_div(min.height, threads.y));
    PopulateMinMaxKernal<<<blocks, threads, 0, stream>>>(apriltag_gpu, min_gpu, max_gpu);
  }

  void GpuApriltagDetector::PopulateThresholdValidGPU(ImageView apriltag, ImageView min, ImageView max, ImageView binarized_apriltag,
                                 cudaStream_t stream){
    ImageView d_apriltag(apriltag);
    ImageViewGPU d_min(min);
    ImageViewGPU d_max(max);
    ImageViewGPU d_binarized_apriltag(binarized_apriltag);
    dim3 threads(8, 8);
    dim3 blocks(cuda::ceil_div(min.width, threads.x), cuda::ceil_div(min.height, threads.y));
    PopulateBinarizedApriltagKernal<<<blocks, threads, 0, stream>>>(d_apriltag, d_min, d_max, d_binarized_apriltag);
  }


  void GpuApriltagDetector::RegisterApriltagViewToGPU(ImageView apriltag){
    CUDA_CHECK(cudaHostRegister(
          apriltag.data, apriltag.stride * apriltag.height * sizeof(uint8_t),
          cudaHostRegisterMapped));
  }


  void GpuApriltagDetector::UnregisterApriltagViewToGPU(ImageView apriltag){
    cudaHostUnregister(apriltag.data);
  }

  void GpuApriltagDetector::SyncStream(){
    CUDA_CHECK(cudaStreamSynchronize(stream_));
  }

}
