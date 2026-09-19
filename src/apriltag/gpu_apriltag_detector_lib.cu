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
}

namespace apriltag{
  void GpuApriltagDetector::PopulateMinMaxGPU(ImageView apriltag, ImageView min, ImageView max){
    ImageViewGPU apriltag_gpu(apriltag);
    ImageViewGPU min_gpu(min);
    ImageViewGPU max_gpu(max);


    dim3 threads(32, 8);
    dim3 blocks(cuda::ceil_div(min.width, threads.x), cuda::ceil_div(min.height, threads.y));
    PopulateMinMaxKernal<<<blocks, threads>>>(apriltag_gpu, min_gpu, max_gpu);
    CUDA_CHECK(cudaDeviceSynchronize());
  }


  void GpuApriltagDetector::RegisterApriltagViewToGPU(ImageView apriltag){
    CUDA_CHECK(cudaHostRegister(
          apriltag.data, apriltag.stride * apriltag.height * sizeof(uint8_t),
          cudaHostRegisterMapped));
  }


  void GpuApriltagDetector::UnregisterApriltagViewToGPU(ImageView apriltag){
    cudaHostUnregister(apriltag.data);
  }
}
