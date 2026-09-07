#include "apriltag/gpu_apriltag_detector_lib.h"

#include "absl/log/check.h"
#include "absl/container/flat_hash_set.h"

#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>

#include <cuda/std/algorithm>
#include <cub/device/device_segmented_sort.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_run_length_encode.cuh>
#include <cub/device/device_scan.cuh>
#include <cub/device/device_select.cuh>
#include <thrust/iterator/counting_iterator.h>

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

  uint8_t min = 255;
  uint8_t max = 0;
#pragma unroll
  for (int row_offset = -1; row_offset <= 1; ++row_offset) {
    const int r = cuda::std::max(0, cuda::std::min(static_cast<int>(min_image_view.height - 1),
                                                   static_cast<int>(row) + row_offset));
#pragma unroll
    for (int col_offset = -1; col_offset <= 1; ++col_offset) {
      const int c = cuda::std::max(0, cuda::std::min(static_cast<int>(min_image_view.width - 1),
                                                     static_cast<int>(col) + col_offset));
      min = cuda::std::min(
          min, Get(min_image_view.data_gpu, min_image_view.stride, r, c));
      max = cuda::std::max(
          max, Get(max_image_view.data_gpu, max_image_view.stride, r, c));
    }
  }
  threshold = (max / 2) + (min / 2);
  valid = (max - min > 8) ? 255 : 0;
}

__global__ void BinarizeKernel(apriltag::ImageView image,
                               apriltag::ImageView threshold,
                               apriltag::ImageView valid, uint8_t* output) {
  const int col = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (col >= image.width || row >= image.height) return;
  const uint8_t t = Get(threshold.data_gpu, threshold.stride, row / 4, col / 4);
  const bool v = Get(valid.data_gpu, valid.stride, row / 4, col / 4) != 0;
  const bool white = Get(image.data_gpu, image.stride, row, col) > t;
  output[static_cast<size_t>(row) * image.width + col] =
      v ? (white ? 255 : 0) : (white ? 177 : 77);
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

void PopulatePreprocessedApriltagGPU(ImageView image, ImageView min, ImageView max,
                                    ImageView threshold, ImageView valid,
                                    uint8_t* device_binarized) {
  CHECK_GT(image.width, 0);
  CHECK_GT(image.height, 0);
  CHECK_EQ(image.width % 4, 0);
  CHECK_EQ(image.height % 4, 0);
  CHECK_GE(image.stride, image.width);
  CHECK(image.data_gpu != nullptr);
  CHECK(device_binarized != nullptr);
  for (const auto& view : {min, max, threshold, valid}) {
    CHECK_EQ(view.width, image.width / 4);
    CHECK_EQ(view.height, image.height / 4);
    CHECK_GE(view.stride, view.width);
    CHECK(view.data_gpu != nullptr);
  }
  constexpr dim3 block(32, 8);
  const dim3 tiles((min.width + block.x - 1) / block.x,
                   (min.height + block.y - 1) / block.y);
  MinMaxKernel<<<tiles, block>>>(image, min, max, min.height, min.width);
  CUDA_CHECK(cudaGetLastError());
  ThresholdValidKernel<<<tiles, block>>>(min, max, threshold, valid);
  CUDA_CHECK(cudaGetLastError());
  const dim3 pixels((image.width + block.x - 1) / block.x,
                    (image.height + block.y - 1) / block.y);
  BinarizeKernel<<<pixels, block>>>(image, threshold, valid, device_binarized);
  CUDA_CHECK(cudaGetLastError());
}

void PopulateSegmentedApriltagDevice(const uint8_t* device_image,
                                     uint32_t* device_labels, int width, int height) {
  CHECK_GE(width, 0);
  CHECK_GE(height, 0);
  if (width == 0 || height == 0) return;
  CHECK(device_image != nullptr);
  CHECK(device_labels != nullptr);
  const size_t pixel_count = static_cast<size_t>(width) * height;
  CHECK_LE(pixel_count, static_cast<size_t>(std::numeric_limits<int>::max() - 255));
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

}

void PopulateSegmentedApriltagGPU(ImageView binarized_apriltag,
                                  ImageView32 segmented_apriltag,
                                  uint8_t* device_image,
                                  uint32_t* device_labels) {
  CHECK_EQ(binarized_apriltag.width, segmented_apriltag.width);
  CHECK_EQ(binarized_apriltag.height, segmented_apriltag.height);
  CHECK_GE(binarized_apriltag.stride, binarized_apriltag.width);
  CHECK_GE(segmented_apriltag.stride, segmented_apriltag.width);
  const int width = binarized_apriltag.width;
  const int height = binarized_apriltag.height;
  CHECK_GE(width, 0);
  CHECK_GE(height, 0);
  if (width == 0 || height == 0) return;
  CUDA_CHECK(cudaMemcpy2D(device_image, width, binarized_apriltag.data,
                          binarized_apriltag.stride, width, height,
                          cudaMemcpyHostToDevice));
  PopulateSegmentedApriltagDevice(device_image, device_labels, width, height);
  CUDA_CHECK(cudaMemcpy2D(segmented_apriltag.data,
                          size_t(segmented_apriltag.stride) * sizeof(uint32_t),
                          device_labels, size_t(width) * sizeof(uint32_t),
                          size_t(width) * sizeof(uint32_t), height,
                          cudaMemcpyDeviceToHost));
}

namespace {

template <typename T>
struct ExtractionBuffer {
  T* data = nullptr;
  size_t capacity = 0;
  ~ExtractionBuffer() { if (data) cudaFree(data); }
  void Reserve(size_t count) {
    if (count <= capacity) return;
    CHECK_LE(count, std::numeric_limits<size_t>::max() / sizeof(T));
    T* replacement = nullptr;
    CUDA_CHECK(cudaMalloc(&replacement, count * sizeof(T)));
    if (data) CUDA_CHECK(cudaFree(data));
    data = replacement;
    capacity = count;
  }
};

__global__ void ExtractBoundaryKeys(const uint32_t* labels, int width,
                                    int stride, int count, uint64_t* keys) {
  const int edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= count) return;
  const int pixel = edge / 2;
  const int row = pixel / (width - 1);
  const int col = pixel % (width - 1);
  const size_t index = static_cast<size_t>(row) * stride + col;
  const uint32_t a = labels[index];
  const uint32_t b = labels[index + ((edge & 1) ? stride : 1)];
  keys[edge] = a && b && a != b
      ? (static_cast<uint64_t>(a > b ? a : b) << 32) | (a < b ? a : b)
      : 0;
}

struct ValidBoundary {
  const uint64_t* keys;
  __device__ bool operator()(int edge) const { return keys[edge] != 0; }
};

struct RetainedBoundaryCount {
  __host__ __device__ bool operator()(int count) const { return count >= 40; }
};

__global__ void GatherBoundaryKeys(const uint64_t* keys, const int* edges,
                                   int count, uint64_t* compact_keys) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) compact_keys[i] = keys[edges[i]];
}

__global__ void MarkRetainedBoundaries(const int* offsets, const int* counts,
                                       uint8_t* flags) {
  const int run = blockIdx.x;
  for (int i = threadIdx.x; i < counts[run]; i += blockDim.x) {
    flags[offsets[run] + i] = counts[run] >= 40;
  }
}

}  // namespace

struct GpuSegmentExtractor::Impl {
  ExtractionBuffer<uint32_t> labels;
  ExtractionBuffer<uint64_t> keys, compact_keys, sorted_keys, unique_keys;
  ExtractionBuffer<int> edges, sorted_edges, retained_edges;
  ExtractionBuffer<int> counts, offsets, retained_counts, result_count;
  ExtractionBuffer<uint8_t> flags, scratch;
  std::vector<int> host_edges, host_counts;

  template <typename Operation>
  void Run(Operation operation) {
    size_t bytes = 0;
    CUDA_CHECK(operation(nullptr, bytes));
    scratch.Reserve(bytes);
    CUDA_CHECK(operation(scratch.data, bytes));
  }

  auto ReadCount() -> int {
    int count = 0;
    CUDA_CHECK(cudaMemcpy(&count, result_count.data, sizeof(count),
                          cudaMemcpyDeviceToHost));
    return count;
  }

  auto Extract(const uint32_t* input, int width, int height, int stride)
      -> std::vector<std::vector<Coord<int>>> {
    CHECK_GE(width, 0);
    CHECK_GE(height, 0);
    CHECK_GE(stride, width);
    if (width < 2 || height < 2) return {};
    CHECK(input != nullptr);
    const size_t slots = size_t{2} * (width - 1) * (height - 1);
    // CUB counts and encoded edge positions use signed int.
    CHECK_LE(slots, static_cast<size_t>(std::numeric_limits<int>::max() - 255));
    const int n = static_cast<int>(slots);
    keys.Reserve(slots);
    edges.Reserve(slots);
    result_count.Reserve(1);
    ExtractBoundaryKeys<<<(n + 255) / 256, 256>>>(
        input, width, stride, n, keys.data);
    CUDA_CHECK(cudaGetLastError());
    Run([&](void* temp, size_t& bytes) {
      return cub::DeviceSelect::If(temp, bytes,
          thrust::counting_iterator<int>(0), edges.data, result_count.data,
          n, ValidBoundary{keys.data});
    });
    const int valid = ReadCount();
    if (valid < 40) return {};
    compact_keys.Reserve(valid);
    sorted_keys.Reserve(valid);
    sorted_edges.Reserve(valid);
    unique_keys.Reserve(valid);
    counts.Reserve(valid);
    offsets.Reserve(valid);
    flags.Reserve(valid);
    retained_edges.Reserve(valid);
    retained_counts.Reserve(valid);
    GatherBoundaryKeys<<<(valid + 255) / 256, 256>>>(
        keys.data, edges.data, valid, compact_keys.data);
    CUDA_CHECK(cudaGetLastError());
    // Stable selection and sorting preserve the CPU scan's order within a pair.
    Run([&](void* temp, size_t& bytes) {
      return cub::DeviceRadixSort::SortPairs(temp, bytes, compact_keys.data,
          sorted_keys.data, edges.data, sorted_edges.data, valid);
    });
    Run([&](void* temp, size_t& bytes) {
      return cub::DeviceRunLengthEncode::Encode(temp, bytes, sorted_keys.data,
          unique_keys.data, counts.data, result_count.data, valid);
    });
    const int runs = ReadCount();
    Run([&](void* temp, size_t& bytes) {
      return cub::DeviceScan::ExclusiveSum(temp, bytes, counts.data,
                                            offsets.data, runs);
    });
    MarkRetainedBoundaries<<<runs, 256>>>(offsets.data, counts.data, flags.data);
    CUDA_CHECK(cudaGetLastError());
    Run([&](void* temp, size_t& bytes) {
      return cub::DeviceSelect::Flagged(temp, bytes, sorted_edges.data,
          flags.data, retained_edges.data, result_count.data, valid);
    });
    const int retained = ReadCount();
    if (retained == 0) return {};
    Run([&](void* temp, size_t& bytes) {
      return cub::DeviceSelect::If(temp, bytes, counts.data,
          retained_counts.data, result_count.data, runs, RetainedBoundaryCount{});
    });
    const int segments_count = ReadCount();
    host_edges.resize(retained);
    host_counts.resize(segments_count);
    CUDA_CHECK(cudaMemcpy(host_edges.data(), retained_edges.data,
                          retained * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_counts.data(), retained_counts.data,
                          segments_count * sizeof(int), cudaMemcpyDeviceToHost));
    std::vector<std::vector<Coord<int>>> segments(segments_count);
    size_t cursor = 0;
    for (int s = 0; s < segments_count; ++s) {
      auto& segment = segments[s];
      segment.reserve(size_t{2} * host_counts[s]);
      for (int i = 0; i < host_counts[s]; ++i) {
        const int edge = host_edges[cursor++];
        const int row = (edge / 2) / (width - 1);
        const int col = (edge / 2) % (width - 1);
        segment.push_back({row + (edge & 1), col + !(edge & 1)});
        segment.push_back({row, col});
      }
    }
    return segments;
  }
};

GpuSegmentExtractor::GpuSegmentExtractor() : impl_(std::make_unique<Impl>()) {}
GpuSegmentExtractor::~GpuSegmentExtractor() = default;
GpuSegmentExtractor::GpuSegmentExtractor(GpuSegmentExtractor&&) noexcept = default;
auto GpuSegmentExtractor::operator=(GpuSegmentExtractor&&) noexcept
    -> GpuSegmentExtractor& = default;

auto GpuSegmentExtractor::ExtractDevice(const uint32_t* labels, int width,
                                        int height, int stride)
    -> std::vector<std::vector<Coord<int>>> {
  return impl_->Extract(labels, width, height, stride);
}

auto GpuSegmentExtractor::DeviceWorkspaceBytes() const -> size_t {
  return impl_->labels.capacity * sizeof(uint32_t) +
      (impl_->keys.capacity + impl_->compact_keys.capacity +
       impl_->sorted_keys.capacity + impl_->unique_keys.capacity) * sizeof(uint64_t) +
      (impl_->edges.capacity + impl_->sorted_edges.capacity +
       impl_->retained_edges.capacity + impl_->counts.capacity +
       impl_->offsets.capacity + impl_->retained_counts.capacity +
       impl_->result_count.capacity) * sizeof(int) +
      impl_->flags.capacity + impl_->scratch.capacity;
}

auto GpuSegmentExtractor::Extract(ImageView32 labels)
    -> std::vector<std::vector<Coord<int>>> {
  CHECK_GE(labels.width, 0);
  CHECK_GE(labels.height, 0);
  CHECK_GE(labels.stride, labels.width);
  if (labels.width < 2 || labels.height < 2) return {};
  CHECK(labels.data != nullptr);
  CHECK_LE(size_t{2} * (labels.width - 1) * (labels.height - 1),
           static_cast<size_t>(std::numeric_limits<int>::max() - 255));
  impl_->labels.Reserve(static_cast<size_t>(labels.width) * labels.height);
  CUDA_CHECK(cudaMemcpy2D(impl_->labels.data, size_t(labels.width) * sizeof(uint32_t),
      labels.data, size_t(labels.stride) * sizeof(uint32_t),
      size_t(labels.width) * sizeof(uint32_t), labels.height, cudaMemcpyHostToDevice));
  return ExtractDevice(impl_->labels.data, labels.width, labels.height, labels.width);
}

auto GetSegments(ImageView32 labels) -> std::vector<std::vector<Coord<int>>> {
  static thread_local GpuSegmentExtractor extractor;
  return extractor.Extract(labels);
}

struct SegmentSortInfo {
  int offset;
  int count;
  int64_t mean_row;
  int64_t mean_col;
};

__global__ void ComputeSortKeysKernel(const Coord<int>* points,
                                      const SegmentSortInfo* segments_info,
                                      double* keys,
                                      int num_segments) {
  const int seg_idx = blockIdx.x;
  if (seg_idx >= num_segments) {
    return;
  }

  const SegmentSortInfo info = segments_info[seg_idx];
  const int offset = info.offset;
  const int count = info.count;
  const double m_row = static_cast<double>(info.mean_row);
  const double m_col = static_cast<double>(info.mean_col);

  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const Coord<int> pt = points[offset + i];
    keys[offset + i] = -atan2(static_cast<double>(pt.row) - m_row,
                              static_cast<double>(pt.col) - m_col);
  }
}

struct GpuSegmentSorter::Impl {
  size_t capacity_points = 0;
  size_t capacity_segments = 0;

  Coord<int>* d_points_in = nullptr;
  Coord<int>* d_points_out = nullptr;
  double* d_keys_in = nullptr;
  double* d_keys_out = nullptr;
  int* d_offsets = nullptr;
  SegmentSortInfo* d_segments_info = nullptr;
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;

  std::vector<Coord<int>> h_points_in;
  std::vector<Coord<int>> h_points_out;
  std::vector<int> h_offsets;
  std::vector<SegmentSortInfo> h_segments_info;

  Impl(size_t initial_points = 65536, size_t initial_segments = 2048) {
    Allocate(initial_points, initial_segments);
  }

  ~Impl() {
    Free();
  }

  void Allocate(size_t points, size_t segments) {
    capacity_points = points;
    capacity_segments = segments;

    CUDA_CHECK(cudaMalloc(&d_points_in, points * sizeof(Coord<int>)));
    CUDA_CHECK(cudaMalloc(&d_points_out, points * sizeof(Coord<int>)));
    CUDA_CHECK(cudaMalloc(&d_keys_in, points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_keys_out, points * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_offsets, (segments + 1) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_segments_info, segments * sizeof(SegmentSortInfo)));

    temp_storage_bytes = 0;
    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(
        nullptr, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_points_in, d_points_out,
        static_cast<int64_t>(points), static_cast<int64_t>(segments),
        d_offsets, d_offsets + 1));
    if (temp_storage_bytes > 0) {
      CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }
  }

  void Free() {
    if (d_points_in) cudaFree(d_points_in);
    if (d_points_out) cudaFree(d_points_out);
    if (d_keys_in) cudaFree(d_keys_in);
    if (d_keys_out) cudaFree(d_keys_out);
    if (d_offsets) cudaFree(d_offsets);
    if (d_segments_info) cudaFree(d_segments_info);
    if (d_temp_storage) cudaFree(d_temp_storage);
    d_points_in = nullptr;
    d_points_out = nullptr;
    d_keys_in = nullptr;
    d_keys_out = nullptr;
    d_offsets = nullptr;
    d_segments_info = nullptr;
    d_temp_storage = nullptr;
    temp_storage_bytes = 0;
    capacity_points = 0;
    capacity_segments = 0;
  }

  void EnsureCapacity(size_t points, size_t segments) {
    if (points > capacity_points || segments > capacity_segments) {
      size_t new_points = std::max(points, capacity_points * 2);
      size_t new_segments = std::max(segments, capacity_segments * 2);
      Free();
      Allocate(new_points, new_segments);
    }
  }

  void Sort(std::vector<std::vector<Coord<int>>>& segments) {
    std::vector<size_t> non_empty_indices;
    non_empty_indices.reserve(segments.size());
    size_t total_points = 0;
    for (size_t i = 0; i < segments.size(); ++i) {
      if (!segments[i].empty()) {
        non_empty_indices.push_back(i);
        total_points += segments[i].size();
      }
    }

    if (non_empty_indices.empty() || total_points == 0) {
      return;
    }

    const int num_segments = static_cast<int>(non_empty_indices.size());
    EnsureCapacity(total_points, num_segments);

    h_points_in.resize(total_points);
    h_offsets.resize(num_segments + 1);
    h_segments_info.resize(num_segments);

    int current_offset = 0;
    for (int seg_idx = 0; seg_idx < num_segments; ++seg_idx) {
      const auto& seg = segments[non_empty_indices[seg_idx]];
      const int seg_size = static_cast<int>(seg.size());
      h_offsets[seg_idx] = current_offset;

      int64_t sum_row = 0;
      int64_t sum_col = 0;
      for (int i = 0; i < seg_size; ++i) {
        h_points_in[current_offset + i] = seg[i];
        sum_row += seg[i].row;
        sum_col += seg[i].col;
      }

      h_segments_info[seg_idx] = SegmentSortInfo{
          .offset = current_offset,
          .count = seg_size,
          .mean_row = sum_row / seg_size,
          .mean_col = sum_col / seg_size,
      };

      current_offset += seg_size;
    }
    h_offsets[num_segments] = current_offset;

    CUDA_CHECK(cudaMemcpy(d_points_in, h_points_in.data(),
                          total_points * sizeof(Coord<int>),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offsets, h_offsets.data(),
                          (num_segments + 1) * sizeof(int),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_segments_info, h_segments_info.data(),
                          num_segments * sizeof(SegmentSortInfo),
                          cudaMemcpyHostToDevice));

    constexpr int block_dim = 128;
    ComputeSortKeysKernel<<<num_segments, block_dim>>>(
        d_points_in, d_segments_info, d_keys_in, num_segments);
    CUDA_CHECK(cudaGetLastError());

    size_t required_temp_bytes = temp_storage_bytes;
    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(
        nullptr, required_temp_bytes,
        d_keys_in, d_keys_out,
        d_points_in, d_points_out,
        static_cast<int64_t>(total_points), static_cast<int64_t>(num_segments),
        d_offsets, d_offsets + 1));

    if (required_temp_bytes > temp_storage_bytes) {
      if (d_temp_storage) {
        CUDA_CHECK(cudaFree(d_temp_storage));
      }
      temp_storage_bytes = required_temp_bytes;
      CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));
    }

    CUDA_CHECK(cub::DeviceSegmentedSort::SortPairs(
        d_temp_storage, temp_storage_bytes,
        d_keys_in, d_keys_out,
        d_points_in, d_points_out,
        static_cast<int64_t>(total_points), static_cast<int64_t>(num_segments),
        d_offsets, d_offsets + 1));
    CUDA_CHECK(cudaGetLastError());

    h_points_out.resize(total_points);
    CUDA_CHECK(cudaMemcpy(h_points_out.data(), d_points_out,
                          total_points * sizeof(Coord<int>),
                          cudaMemcpyDeviceToHost));

    for (int seg_idx = 0; seg_idx < num_segments; ++seg_idx) {
      auto& seg = segments[non_empty_indices[seg_idx]];
      const int start_idx = h_offsets[seg_idx];
      const int end_idx = h_offsets[seg_idx + 1];
      auto* start_ptr = h_points_out.data() + start_idx;
      auto* end_ptr = h_points_out.data() + end_idx;

      // Equal angular keys need not group identical coordinates together.
      absl::flat_hash_set<Coord<int>> seen;
      seen.reserve(end_idx - start_idx);
      seg.clear();
      for (auto* point = start_ptr; point != end_ptr; ++point) {
        if (seen.insert(*point).second) seg.push_back(*point);
      }
    }
  }
};

GpuSegmentSorter::GpuSegmentSorter() : impl_(std::make_unique<Impl>()) {}
GpuSegmentSorter::~GpuSegmentSorter() = default;
GpuSegmentSorter::GpuSegmentSorter(GpuSegmentSorter&&) noexcept = default;
auto GpuSegmentSorter::operator=(GpuSegmentSorter&&) noexcept
    -> GpuSegmentSorter& = default;

void GpuSegmentSorter::Sort(std::vector<std::vector<Coord<int>>>& segments) {
  impl_->Sort(segments);
}

void SortSegmentsGPU(std::vector<std::vector<Coord<int>>>& segments) {
  static thread_local GpuSegmentSorter sorter;
  sorter.Sort(segments);
}

__device__ inline float DeviceGetBlackWhiteThreshold(
    const uint8_t* __restrict__ image, int stride, int width, int height,
    const BitLocation& bit_location) {
  float white = 0.0f;
  int white_count = 0;

  for (int i = 0; i < 10; i++) {
    const Coord<int> pts[4] = {
        bit_location[0][i], bit_location[9][i],
        bit_location[i][0], bit_location[i][9]};
    for (int k = 0; k < 4; k++) {
      const int r = pts[k].row;
      const int c = pts[k].col;
      if (r >= 0 && r < height && c >= 0 && c < width) {
        white += static_cast<float>(image[r * stride + c]);
        white_count++;
      }
    }
  }
  white = (white_count > 0) ? (white / static_cast<float>(white_count)) : 255.0f;

  float black = 0.0f;
  int black_count = 0;
  for (int i = 1; i < 9; i++) {
    const Coord<int> pts[4] = {
        bit_location[1][i], bit_location[8][i],
        bit_location[i][1], bit_location[i][8]};
    for (int k = 0; k < 4; k++) {
      const int r = pts[k].row;
      const int c = pts[k].col;
      if (r >= 0 && r < height && c >= 0 && c < width) {
        black += static_cast<float>(image[r * stride + c]);
        black_count++;
      }
    }
  }
  black = (black_count > 0) ? (black / static_cast<float>(black_count)) : 0.0f;

  return (white + black) * 0.5f;
}

__device__ inline uint64_t DeviceExtractCodeword(
    const uint8_t* __restrict__ image, int stride, int width, int height,
    const BitLocation& bit_location,
    const uint32_t* __restrict__ bit_x,
    const uint32_t* __restrict__ bit_y,
    uint32_t nbits,
    float thresh) {
  uint64_t code = 0;
  for (uint32_t j = 0; j < nbits; j++) {
    const uint32_t x = bit_x[j];
    const uint32_t y = bit_y[j];
    code <<= 1;
    const int r = bit_location[y + 1][x + 1].row;
    const int c = bit_location[y + 1][x + 1].col;
    if (r >= 0 && r < height && c >= 0 && c < width) {
      if (static_cast<float>(image[r * stride + c]) > thresh) {
        code |= 1ULL;
      }
    }
  }
  return code;
}

__device__ inline bool DeviceMatchCodeword(
    uint64_t code,
    int num_codes,
    const uint64_t* __restrict__ target_codes,
    const int* __restrict__ target_ids,
    int& best_id,
    int& best_rotation,
    int& best_hamming) {
  constexpr int nbits = 36;
  constexpr int shift = 9;
  constexpr uint64_t mask = (1ULL << nbits) - 1;

  for (int j = 0; j < 4; j++) {
    for (int k = 0; k < num_codes; k++) {
      const int hamming = __popcll(code ^ target_codes[k]);
      if (hamming < best_hamming) {
        best_hamming = hamming;
        best_id = target_ids[k];
        best_rotation = j;
        if (hamming == 0) {
          return true;
        }
      }
    }
    if (best_hamming == 0) {
      return true;
    }
    code = ((code << shift) | (code >> (nbits - shift))) & mask;
  }
  return best_hamming <= 2;
}

__global__ void DecodeTagIdsKernel(
    const BitLocation* __restrict__ bit_locations,
    int num_quads,
    const uint8_t* __restrict__ image,
    int stride, int width, int height,
    const uint32_t* __restrict__ bit_x,
    const uint32_t* __restrict__ bit_y,
    uint32_t nbits,
    const uint64_t* __restrict__ target_codes,
    const int* __restrict__ target_ids,
    int num_target_codes,
    int* __restrict__ out_tag_ids,
    int* __restrict__ out_rotations) {
  const int quad_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (quad_idx >= num_quads) {
    return;
  }

  const BitLocation bit_loc = bit_locations[quad_idx];
  if (bit_loc[0][0].row == 0 && bit_loc[0][0].col == 0 &&
      bit_loc[9][9].row == 0 && bit_loc[9][9].col == 0) {
    out_tag_ids[quad_idx] = -1;
    out_rotations[quad_idx] = -1;
    return;
  }

  const float threshold =
      DeviceGetBlackWhiteThreshold(image, stride, width, height, bit_loc);

  int best_id = -1;
  int best_rotation = -1;
  int best_hamming = 3;

  // Pass 1: base threshold
  uint64_t code = DeviceExtractCodeword(
      image, stride, width, height, bit_loc, bit_x, bit_y, nbits, threshold);
  DeviceMatchCodeword(code, num_target_codes, target_codes, target_ids,
                      best_id, best_rotation, best_hamming);

  // Passes 2-5: retry with deltas {-8, +8, -16, +16} if not decoded
  if (best_id == -1) {
    const float deltas[4] = {-8.0f, 8.0f, -16.0f, 16.0f};
    #pragma unroll
    for (int d = 0; d < 4; d++) {
      code = DeviceExtractCodeword(image, stride, width, height, bit_loc, bit_x,
                                   bit_y, nbits, threshold + deltas[d]);
      DeviceMatchCodeword(code, num_target_codes, target_codes, target_ids,
                          best_id, best_rotation, best_hamming);
      if (best_id != -1) {
        break;
      }
    }
  }

  out_tag_ids[quad_idx] = best_id;
  out_rotations[quad_idx] = best_rotation;
}

struct GpuTagIdDecoder::Impl {
  size_t capacity_quads = 0;

  BitLocation* d_bit_locations = nullptr;
  int* d_out_tag_ids = nullptr;
  int* d_out_rotations = nullptr;

  uint32_t* d_bit_x = nullptr;
  uint32_t* d_bit_y = nullptr;
  uint32_t nbits = 0;

  uint64_t* d_target_codes = nullptr;
  int* d_target_ids = nullptr;
  int num_target_codes = 0;
  size_t capacity_target_codes = 0;

  std::vector<int> h_out_tag_ids;
  std::vector<int> h_out_rotations;

  Impl(size_t initial_quads = 1024) {
    Allocate(initial_quads);
  }

  ~Impl() {
    Free();
  }

  void Allocate(size_t quads) {
    capacity_quads = quads;
    CUDA_CHECK(cudaMalloc(&d_bit_locations, quads * sizeof(BitLocation)));
    CUDA_CHECK(cudaMalloc(&d_out_tag_ids, quads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_out_rotations, quads * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bit_x, 64 * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_bit_y, 64 * sizeof(uint32_t)));
  }

  void Free() {
    if (d_bit_locations) cudaFree(d_bit_locations);
    if (d_out_tag_ids) cudaFree(d_out_tag_ids);
    if (d_out_rotations) cudaFree(d_out_rotations);
    if (d_bit_x) cudaFree(d_bit_x);
    if (d_bit_y) cudaFree(d_bit_y);
    if (d_target_codes) cudaFree(d_target_codes);
    if (d_target_ids) cudaFree(d_target_ids);
    d_bit_locations = nullptr;
    d_out_tag_ids = nullptr;
    d_out_rotations = nullptr;
    d_bit_x = nullptr;
    d_bit_y = nullptr;
    d_target_codes = nullptr;
    d_target_ids = nullptr;
    capacity_target_codes = 0;
    capacity_quads = 0;
  }

  void EnsureCapacity(size_t quads) {
    if (quads > capacity_quads) {
      if (d_bit_locations) cudaFree(d_bit_locations);
      if (d_out_tag_ids) cudaFree(d_out_tag_ids);
      if (d_out_rotations) cudaFree(d_out_rotations);
      capacity_quads = std::max(quads, capacity_quads * 2);
      CUDA_CHECK(cudaMalloc(&d_bit_locations, capacity_quads * sizeof(BitLocation)));
      CUDA_CHECK(cudaMalloc(&d_out_tag_ids, capacity_quads * sizeof(int)));
      CUDA_CHECK(cudaMalloc(&d_out_rotations, capacity_quads * sizeof(int)));
    }
  }

  void SetTargetCodes(apriltag_family_t* family,
                      const std::vector<int>& target_tag_ids) {
    if (!family) {
      return;
    }
    nbits = family->nbits;
    CUDA_CHECK(cudaMemcpy(d_bit_x, family->bit_x, nbits * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bit_y, family->bit_y, nbits * sizeof(uint32_t),
                          cudaMemcpyHostToDevice));

    std::vector<uint64_t> codes;
    std::vector<int> ids;
    if (!target_tag_ids.empty()) {
      codes.reserve(target_tag_ids.size());
      ids.reserve(target_tag_ids.size());
      for (int id : target_tag_ids) {
        if (id >= 0 && static_cast<uint32_t>(id) < family->ncodes) {
          codes.push_back(family->codes[id]);
          ids.push_back(id);
        }
      }
    } else {
      codes.reserve(family->ncodes);
      ids.reserve(family->ncodes);
      for (uint32_t i = 0; i < family->ncodes; ++i) {
        codes.push_back(family->codes[i]);
        ids.push_back(static_cast<int>(i));
      }
    }
    CHECK_LE(codes.size(), static_cast<size_t>(std::numeric_limits<int>::max()));
    num_target_codes = static_cast<int>(codes.size());
    if (num_target_codes == 0) return;
    if (codes.size() > capacity_target_codes) {
      if (d_target_codes) CUDA_CHECK(cudaFree(d_target_codes));
      if (d_target_ids) CUDA_CHECK(cudaFree(d_target_ids));
      capacity_target_codes = std::max(codes.size(), capacity_target_codes * 2);
      CUDA_CHECK(cudaMalloc(&d_target_codes,
                            capacity_target_codes * sizeof(uint64_t)));
      CUDA_CHECK(cudaMalloc(&d_target_ids,
                            capacity_target_codes * sizeof(int)));
    }
    CUDA_CHECK(cudaMemcpy(d_target_codes, codes.data(),
                          num_target_codes * sizeof(uint64_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target_ids, ids.data(),
                          num_target_codes * sizeof(int),
                          cudaMemcpyHostToDevice));
  }

  auto Decode(const std::vector<BitLocation>& bit_locations,
              ImageView apriltag)
      -> std::pair<std::vector<int>, std::vector<int>> {
    const int num_quads = static_cast<int>(bit_locations.size());
    if (num_quads == 0) {
      return {{}, {}};
    }

    CHECK(apriltag.data_gpu != nullptr);
    EnsureCapacity(num_quads);

    CUDA_CHECK(cudaMemcpy(d_bit_locations, bit_locations.data(),
                          num_quads * sizeof(BitLocation),
                          cudaMemcpyHostToDevice));

    constexpr int block_size = 64;
    const int grid_size = (num_quads + block_size - 1) / block_size;
    DecodeTagIdsKernel<<<grid_size, block_size>>>(
        d_bit_locations, num_quads,
        apriltag.data_gpu, apriltag.stride, apriltag.width, apriltag.height,
        d_bit_x, d_bit_y, nbits,
        d_target_codes, d_target_ids, num_target_codes,
        d_out_tag_ids, d_out_rotations);
    CUDA_CHECK(cudaGetLastError());

    h_out_tag_ids.resize(num_quads);
    h_out_rotations.resize(num_quads);
    CUDA_CHECK(cudaMemcpy(h_out_tag_ids.data(), d_out_tag_ids,
                          num_quads * sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_out_rotations.data(), d_out_rotations,
                          num_quads * sizeof(int), cudaMemcpyDeviceToHost));

    return {h_out_tag_ids, h_out_rotations};
  }
};

GpuTagIdDecoder::GpuTagIdDecoder() : impl_(std::make_unique<Impl>()) {}
GpuTagIdDecoder::~GpuTagIdDecoder() = default;
GpuTagIdDecoder::GpuTagIdDecoder(GpuTagIdDecoder&&) noexcept = default;
auto GpuTagIdDecoder::operator=(GpuTagIdDecoder&&) noexcept
    -> GpuTagIdDecoder& = default;

void GpuTagIdDecoder::SetTargetCodes(
    apriltag_family_t* family, const std::vector<int>& target_tag_ids) {
  impl_->SetTargetCodes(family, target_tag_ids);
}

auto GpuTagIdDecoder::Decode(const std::vector<BitLocation>& bit_locations,
                             ImageView apriltag)
    -> std::pair<std::vector<int>, std::vector<int>> {
  return impl_->Decode(bit_locations, apriltag);
}

auto GetTagIdsGPU(const std::vector<BitLocation>& bit_locations,
                  ImageView apriltag,
                  apriltag_family_t* family,
                  const std::vector<int>& target_tag_ids)
    -> std::pair<std::vector<int>, std::vector<int>> {
  static thread_local GpuTagIdDecoder decoder;
  decoder.SetTargetCodes(family, target_tag_ids);
  return decoder.Decode(bit_locations, apriltag);
}

}  // namespace apriltag
