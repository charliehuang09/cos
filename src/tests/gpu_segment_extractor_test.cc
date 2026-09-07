#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "apriltag/gpu_apriltag_detector_lib.h"

namespace {
using Segments = std::vector<std::vector<apriltag::Coord<int>>>;

void Canonicalize(Segments& segments) {
  std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
        [](const auto& x, const auto& y) {
          return x.row != y.row ? x.row < y.row : x.col < y.col;
        });
  });
}

class ExtractorTest : public testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "CUDA device required";
    }
  }

  void Compare(std::vector<uint32_t>& pixels, int width, int height, int stride) {
    apriltag::ImageView32 view{pixels.data(), stride, height, width};
    auto expected = apriltag::GetSegmentsCPU(view);
    auto actual = extractor.Extract(view);
    Canonicalize(expected);
    Canonicalize(actual);
    EXPECT_EQ(actual, expected);

    if (width < 2 || height < 2) return;
    uint32_t* device = nullptr;
    ASSERT_EQ(cudaMalloc(&device, pixels.size() * sizeof(uint32_t)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device, pixels.data(), pixels.size() * sizeof(uint32_t),
                         cudaMemcpyHostToDevice), cudaSuccess);
    auto direct = extractor.ExtractDevice(device, width, height, stride);
    EXPECT_EQ(cudaFree(device), cudaSuccess);
    Canonicalize(direct);
    EXPECT_EQ(direct, expected);
  }

  apriltag::GpuSegmentExtractor extractor;
};

TEST_F(ExtractorTest, EmptyTinyAndUniform) {
  for (int size : {0, 1, 2, 32}) {
    for (uint32_t label : {0u, 1u, std::numeric_limits<uint32_t>::max()}) {
      std::vector<uint32_t> pixels(size * size, label);
      Compare(pixels, size, size, size);
    }
  }
}

TEST_F(ExtractorTest, ThresholdAndPaddedStride) {
  for (int edges : {39, 40, 41}) {
    const int height = edges + 1;
    std::vector<uint32_t> pixels(height * 5, 12345);
    for (int row = 0; row < height; ++row) {
      pixels[row * 5] = 1;
      pixels[row * 5 + 1] = std::numeric_limits<uint32_t>::max();
    }
    Compare(pixels, 2, height, 5);
    auto result = extractor.Extract({pixels.data(), 5, height, 2});
    ASSERT_EQ(result.size(), edges >= 40 ? 1u : 0u);
    if (!result.empty()) {
      EXPECT_EQ(result[0].size(), 2u * edges);
    }
  }
}

TEST_F(ExtractorTest, DuplicatePointsAndExcludedBorderStarts) {
  const int width = 65;
  std::vector<uint32_t> pixels(width * width);
  for (int row = 0; row < width; ++row) {
    for (int col = 0; col < width; ++col) {
      pixels[row * width + col] = 1 + (row + col) % 2;
    }
  }
  Compare(pixels, width, width, width);
  auto result = extractor.Extract({pixels.data(), width, width, width});
  ASSERT_EQ(result.size(), 1u);
  EXPECT_EQ(result[0].size(), 4u * (width - 1) * (width - 1));
  std::fill(pixels.begin(), pixels.end(), 0);
  for (int i = 0; i < width; ++i) {
    pixels[(width - 1) * width + i] = 1 + i % 2;
    pixels[i * width + width - 1] = 1 + i % 2;
  }
  Compare(pixels, width, width, width);
  EXPECT_TRUE(extractor.Extract({pixels.data(), width, width, width}).empty());
}

TEST_F(ExtractorTest, RandomizedAndChangingCapacity) {
  std::mt19937 random(73);
  for (int width : {17, 128, 3, 256, 64, 2}) {
    for (uint32_t label_count : {3u, 31u, 100000u}) {
      const int height = width + 7;
      const int stride = width + 3;
      std::vector<uint32_t> pixels(stride * height);
      for (auto& pixel : pixels) pixel = random() % label_count;
      Compare(pixels, width, height, stride);
    }
  }
}
}  // namespace
