#include <algorithm>
#include <array>
#include <random>
#include <future>
#include <memory>
#include <set>

#include <tag36h11.h>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "apriltag/gpu_apriltag_detector.h"

namespace {
class PipelineTest : public testing::Test {
 protected:
  void SetUp() override {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
      GTEST_SKIP() << "CUDA device required";
    }
  }
};

struct MappedImage {
  apriltag::ImageView view{};
  ~MappedImage() { if (view.data) cudaFreeHost(view.data); }
  void Allocate(int width, int height, int stride) {
    ASSERT_EQ(cudaHostAlloc(&view.data, size_t(stride) * height,
                            cudaHostAllocMapped), cudaSuccess);
    view.width = width;
    view.height = height;
    view.stride = stride;
    view.EnableGpu();
  }
};

TEST_F(PipelineTest, PreprocessingMatchesCpuWithPaddedStride) {
  std::mt19937 random(191);
  for (int width : {4, 12, 128}) {
    const int height = width + 4;
    const int stride = width + 13;
    MappedImage image;
    image.Allocate(width, height, stride);
    std::array<MappedImage, 4> gpu;
    std::array<std::vector<uint8_t>, 4> cpu;
    std::array<apriltag::ImageView, 4> cpu_views;
    for (int i = 0; i < 4; ++i) {
      gpu[i].Allocate(width / 4, height / 4, width / 4 + 3);
      cpu[i].resize(size_t(width / 4) * (height / 4));
      cpu_views[i] = {.data = cpu[i].data(), .stride = width / 4,
                       .height = height / 4, .width = width / 4};
    }
    uint8_t* device_binary = nullptr;
    ASSERT_EQ(cudaMalloc(&device_binary, width * height), cudaSuccess);
    std::vector<uint8_t> binary(width * height), expected(width * height);
    apriltag::ImageView expected_view{.data = expected.data(), .stride = width,
                                      .height = height, .width = width};
    for (int pattern = 0; pattern < 4; ++pattern) {
      for (int i = 0; i < stride * height; ++i) {
        image.view.data[i] = pattern == 0 ? 0 : pattern == 1 ? 255 :
            pattern == 2 ? 120 + random() % 9 : random() % 256;
      }
      apriltag::PopulateMinMax(image.view, cpu_views[0], cpu_views[1]);
      apriltag::PopulateThresholdValid(cpu_views[0], cpu_views[1], cpu_views[2], cpu_views[3]);
      apriltag::PopulateBinarizedApriltag(cpu_views[2], cpu_views[3], image.view, expected_view);
      apriltag::PopulatePreprocessedApriltagGPU(image.view, gpu[0].view, gpu[1].view,
                                               gpu[2].view, gpu[3].view, device_binary);
      ASSERT_EQ(cudaMemcpy(binary.data(), device_binary, binary.size(),
                           cudaMemcpyDeviceToHost), cudaSuccess);
      EXPECT_EQ(binary, expected);
      for (int i = 0; i < 4; ++i) {
        for (int row = 0; row < height / 4; ++row) {
          for (int col = 0; col < width / 4; ++col) {
            EXPECT_EQ(gpu[i].view(row, col), cpu_views[i](row, col));
          }
        }
      }
    }
    EXPECT_EQ(cudaFree(device_binary), cudaSuccess);
  }
}

TEST_F(PipelineTest, RepeatedMappedAndHostInputsHaveIdenticalLabels) {
  constexpr int width = 128, height = 80, stride = 137;
  MappedImage image;
  image.Allocate(width, height, stride);
  std::mt19937 random(81);
  apriltag::GPUApriltagDetector detector(width, height);
  for (int frame = 0; frame < 4; ++frame) {
    for (int i = 0; i < stride * height; ++i) image.view.data[i] = random() % 256;
    auto host_view = image.view;
    host_view.data_gpu = nullptr;
    detector.Detect(host_view, frame % 2 == 0);
    auto labels = detector.GetSegmentedApriltagView();
    const std::vector<uint32_t> host_labels(labels.data, labels.data + width * height);
    detector.Detect(image.view, frame % 2 != 0);
    labels = detector.GetSegmentedApriltagView();
    EXPECT_EQ(host_labels, std::vector<uint32_t>(labels.data, labels.data + width * height));
  }
}

TEST(GeometryTest, FourPointWindowWrapsAtSegmentEnd) {
  std::vector<std::vector<apriltag::Coord<int>>> segments{
      {{0, 0}, {0, 2}, {2, 2}, {2, 0}}};
  EXPECT_EQ(apriltag::GetMses(segments),
             (std::vector<std::vector<float>>{{1, 1, 1, 1}}));
}

TEST(GeometryTest, ParallelMatchesSequentialIncludingConcurrentCalls) {
  for (int count : {0, 1, 63, 64, 200}) {
    std::vector<std::vector<apriltag::Coord<int>>> segments(count);
    for (int s = 0; s < count; ++s) {
      const int side = 20 + s % 51;
      for (int i = 0; i < side; ++i) segments[s].push_back({s, i + s});
      for (int i = 0; i < side; ++i) segments[s].push_back({s + i, side + s});
      for (int i = side; i > 0; --i) segments[s].push_back({s + side, i + s});
      for (int i = side; i > 0; --i) segments[s].push_back({s + i, s});
      if (s % 13 == 0) segments[s].clear();
      if (s % 17 == 1) segments[s].resize(3);
    }
    const auto expected = apriltag::GetCandidatesQuadCorners(segments, apriltag::GetMses(segments));
    auto first = std::async(std::launch::async, [&] {
      return apriltag::GetCandidatesQuadCornersParallel(segments);
    });
    auto second = std::async(std::launch::async, [&] {
      return apriltag::GetCandidatesQuadCornersParallel(segments);
    });
    for (auto* future : {&first, &second}) {
      const auto actual = future->get();
      ASSERT_EQ(actual.size(), expected.size());
      for (size_t i = 0; i < actual.size(); ++i) {
        EXPECT_EQ(actual[i].corners, expected[i].corners);
      }
    }
  }
}
// Alternating points on the same ray keep duplicates apart in an angle-only sort.
void CheckBoundaryDeduplication(bool gpu) {
  std::vector<apriltag::Coord<int>> points;
  for (int i = 0; i < 40; ++i) {
    for (int sign : {-1, 1}) {
      points.push_back({sign, sign});
      points.push_back({2 * sign, 2 * sign});
      points.push_back({sign, sign});
    }
  }
  std::mt19937 random(72);
  for (int trial = 0; trial < 10; ++trial) {
    std::shuffle(points.begin(), points.end(), random);
    std::vector<std::vector<apriltag::Coord<int>>> segments{points, {}, {{3, 4}}};
    if (gpu) apriltag::SortSegmentsGPU(segments);
    else apriltag::SortSegmentsCPU(segments);
    ASSERT_EQ(segments[0].size(), 4u);
    std::set<std::pair<int, int>> coordinates;
    for (const auto& point : segments[0]) coordinates.emplace(point.row, point.col);
    EXPECT_EQ(coordinates, (std::set<std::pair<int, int>>{
        {-2, -2}, {-1, -1}, {1, 1}, {2, 2}}));
    EXPECT_TRUE(segments[1].empty());
    EXPECT_EQ(segments[2], (std::vector<apriltag::Coord<int>>{{3, 4}}));
  }
}

TEST(GeometryTest, CpuRemovesNonAdjacentDuplicateCoordinates) {
  CheckBoundaryDeduplication(false);
}

TEST_F(PipelineTest, GpuRemovesNonAdjacentDuplicateCoordinates) {
  CheckBoundaryDeduplication(true);
}

apriltag::BitLocation RenderCode(apriltag_family_t* family, int id,
                                 apriltag::ImageView image) {
  apriltag::BitLocation locations{};
  for (int row = 0; row < 10; ++row) {
    for (int col = 0; col < 10; ++col) {
      locations[row][col] = {row, col};
      image(row, col) = row == 0 || row == 9 || col == 0 || col == 9 ? 255 : 0;
    }
  }
  for (uint32_t bit = 0; bit < family->nbits; ++bit) {
    image(family->bit_y[bit] + 1, family->bit_x[bit] + 1) =
        (family->codes[id] >> (family->nbits - bit - 1)) & 1 ? 255 : 0;
  }
  return locations;
}

TEST(DecoderTest, CpuPreservesFilteredEmptyTargets) {
  std::unique_ptr<apriltag_family_t, decltype(&tag36h11_destroy)>
      family(tag36h11_create(), tag36h11_destroy);
  std::vector<uint8_t> pixels(100);
  apriltag::ImageView image{.data = pixels.data(), .stride = 10, .height = 10, .width = 10};
  std::vector<apriltag::BitLocation> locations{RenderCode(family.get(), 0, image)};
  for (const auto& targets : std::vector<std::vector<int>>{
           {}, {-1}, {static_cast<int>(family->ncodes)}, {-1, 0}, {1}}) {
    const auto result = apriltag::GetTagIdsCPU(locations, image, family.get(), targets);
    const bool matches = targets.empty() || targets == std::vector<int>{-1, 0};
    EXPECT_EQ(result.first, std::vector<int>{matches ? 0 : -1});
    EXPECT_EQ(result.second, std::vector<int>{matches ? 0 : -1});
  }
}

TEST_F(PipelineTest, DecoderGrowsTargetsAndMatchesCpuFiltering) {
  std::unique_ptr<apriltag_family_t, decltype(&tag36h11_destroy)>
      small_family(tag36h11_create(), tag36h11_destroy);
  // The SDK only supplies tag36h11; extend its code table to exercise a
  // default family selection larger than the former 1024-entry capacity.
  std::vector<uint64_t> codes(2320, small_family->codes[0]);
  codes.back() = small_family->codes[small_family->ncodes - 1];
  apriltag_family_t large_family = *small_family;
  large_family.codes = codes.data();
  large_family.ncodes = codes.size();
  MappedImage image;
  image.Allocate(10, 10, 10);
  apriltag::GpuTagIdDecoder decoder;
  for (auto* family : {small_family.get(), &large_family, small_family.get()}) {
    const int id = family->ncodes - 1;
    std::vector<apriltag::BitLocation> locations{RenderCode(family, id, image.view)};
    for (const auto& targets : std::vector<std::vector<int>>{
             {-1}, {id}, {}, std::vector<int>(5000, id),
             {static_cast<int>(family->ncodes)}, {-1, id}, {}}) {
      decoder.SetTargetCodes(family, targets);
      EXPECT_EQ(decoder.Decode(locations, image.view),
                apriltag::GetTagIdsCPU(locations, image.view, family, targets));
    }
  }
}
}  // namespace
