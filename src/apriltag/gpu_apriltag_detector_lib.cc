#include "apriltag/gpu_apriltag_detector_lib.h"
#include <cuda_runtime.h>
#include <tag36h11.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "control_loop/timer.h"

#include <climits>
#include <cstdint>
#include <opencv2/opencv.hpp>

namespace {
[[gnu::always_inline]]
inline auto Mix(uint32_t value) -> uint32_t {
  value ^= value >> 16;
  value *= 0x7FEB352Du;
  value ^= value >> 15;
  value *= 0x846CA68Bu;
  value ^= value >> 16;
  return value;
}

inline void PopulateColor(uint32_t id, uint8_t& r, uint8_t& g, uint8_t& b) {
  if (id == UINT32_MAX) {
    r = 0;
    g = 0;
    b = 0;
    return;
  }

  const uint32_t value = Mix(id);

  r = static_cast<uint8_t>(value);
  g = static_cast<uint8_t>(value >> 8);
  b = static_cast<uint8_t>(value >> 16);
}

[[gnu::always_inline]]
auto inline SolveQuadratic(float a, float b, float c)
    -> std::pair<float, float> {
  float determinant = std::sqrt((b * b) - (4 * a * c));
  return {(-b + determinant) / (2 * a), (-b - determinant) / (2 * a)};
}

[[gnu::always_inline]]
void inline GetMinMax(apriltag::ImageView<uint8_t> apriltag, int row, int col,
                      uint8_t& min, uint8_t& max) {
  row *= 4;
  col *= 4;
  min = apriltag(row, col);
  max = apriltag(row, col);
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      min = std::min(min, (apriltag(row + i, col + j)));
      max = std::max(max, (apriltag(row + i, col + j)));
    }
  }
}

[[gnu::always_inline]]
void inline ApplyThreshold(uint8_t threshold, uint8_t valid,
                           apriltag::ImageView<uint8_t> apriltag,
                           apriltag::ImageView<uint8_t> binarized_apriltag,
                           int row, int col) {
  row *= 4;
  col *= 4;
  if (valid == 0) {
    for (int i = row; i < row + 4; i++) {
      for (int j = col; j < col + 4; j++) {
        binarized_apriltag(i, j) =
            (apriltag(i, j) > threshold) ? (255 / 2) + 50 : (255 / 2) - 50;
      }
    }
  } else {
    for (int i = row; i < row + 4; i++) {
      for (int j = col; j < col + 4; j++) {
        binarized_apriltag(i, j) = (apriltag(i, j) > threshold) ? 255 : 0;
      }
    }
  }
}

[[maybe_unused]]
void PrintCode(const apriltag::BitLocation& bit_location,
               apriltag::ImageView<uint8_t> binarized_apriltag) {
  for (int i = 1; i <= 8; i++) {
    for (int j = 1; j <= 8; j++) {
      std::cout << static_cast<int>(binarized_apriltag(bit_location[i][j].row,
                                                       bit_location[i][j].col) >
                                    255 / 2)
                << " ";
    }
    std::cout << "\n";
  }
  std::cout << "----------------------------\n";
}

}  // namespace

namespace apriltag {

GpuApriltagDetector::GpuApriltagDetector(int width, int height)
    : width_(width),
      height_(height),
      family_(tag36h11_create(), tag36h11_destroy),
      segment_resource_([width, height] {
        CHECK_GT(width, 0);
        CHECK_GT(height, 0);
        const size_t pixels = static_cast<size_t>(width) * height;
        CHECK_LE(pixels, SIZE_MAX / sizeof(Coord<int>));
        return std::pmr::pool_options{.largest_required_pool_block =
                                          std::max(pixels, size_t{2048}) *
                                          sizeof(Coord<int>)};
      }()) {
  CHECK_GT(width_, 0);
  CHECK_GT(height_, 0);
  CHECK_EQ(width_ % 4, 0);
  CHECK_EQ(height_ % 4, 0);
  CHECK(cudaStreamCreate(&stream_) == cudaSuccess);

  const size_t pixels = static_cast<size_t>(width_) * height_;
  visited_segment_ids_.resize(pixels);

  CHECK(cudaMallocManaged(reinterpret_cast<void**>(&graph_input_buffer_),
                          pixels * sizeof(uint8_t)) == cudaSuccess);
  graph_input_view_ = {.data = graph_input_buffer_,
                       .stride = width_,
                       .height = height_,
                       .width = width_};

  {
    cudaMallocManaged(reinterpret_cast<void**>(&max_buffer_),
                      (pixels / 16) * sizeof(uint8_t));
    std::memset(max_buffer_, 0, (pixels / 16) * sizeof(uint8_t));
    cudaMallocManaged(reinterpret_cast<void**>(&min_buffer_),
                      (pixels / 16) * sizeof(uint8_t));
    std::memset(min_buffer_, 0, (pixels / 16) * sizeof(uint8_t));
  }
  std::memset(min_buffer_, 0, (pixels / 16) * sizeof(uint8_t));
  threshold_buffer_ =
      static_cast<uint8_t*>(std::malloc((pixels / 16) * sizeof(uint8_t)));
  std::memset(threshold_buffer_, 0, (pixels / 16) * sizeof(uint8_t));
  valid_buffer_ =
      static_cast<uint8_t*>(std::malloc((pixels / 16) * sizeof(uint8_t)));
  std::memset(valid_buffer_, 0, (pixels / 16) * sizeof(uint8_t));
  {
    cudaMallocManaged(reinterpret_cast<void**>(&binarized_apriltag_buffer_),
                      pixels * sizeof(uint8_t));
    std::memset(binarized_apriltag_buffer_, 0, pixels * sizeof(uint8_t));
    cudaMallocManaged(reinterpret_cast<void**>(&segmented_apriltag_buffer_),
                      pixels * sizeof(uint32_t));
    std::memset(segmented_apriltag_buffer_, 0, pixels * sizeof(uint32_t));
  }
  boundary_segmented_apriltag_buffer_ =
      static_cast<uint32_t*>(std::malloc(pixels * sizeof(uint32_t)));
  std::memset(boundary_segmented_apriltag_buffer_, 0,
              pixels * sizeof(uint32_t));
  sorted_boundary_segmented_apriltag_buffer_ =
      static_cast<uint8_t*>(std::malloc(pixels * sizeof(uint8_t)));
  std::memset(sorted_boundary_segmented_apriltag_buffer_, 0,
              pixels * sizeof(uint8_t));
  candidate_quad_corners_apriltag_buffer_ =
      static_cast<uint8_t*>(std::malloc(pixels * sizeof(uint8_t)));
  std::memset(candidate_quad_corners_apriltag_buffer_, 0,
              pixels * sizeof(uint8_t));
  quad_apriltag_buffer_ =
      static_cast<uint8_t*>(std::malloc(pixels * sizeof(uint8_t)));
  std::memset(quad_apriltag_buffer_, 0, pixels * sizeof(uint8_t));
  bit_locations_apriltag_buffer_ =
      static_cast<uint32_t*>(std::malloc(pixels * sizeof(uint32_t)));
  std::memset(bit_locations_apriltag_buffer_, 0, pixels * sizeof(uint32_t));
  refined_points_apriltag_buffer_ =
      static_cast<uint8_t*>(std::malloc(pixels * sizeof(uint8_t)));
  std::memset(refined_points_apriltag_buffer_, 0, pixels * sizeof(uint8_t));
  debug_r_buffer_ =
      static_cast<uint8_t*>(std::malloc(pixels * sizeof(uint8_t)));
  std::memset(debug_r_buffer_, 0, pixels * sizeof(uint8_t));
  debug_g_buffer_ =
      static_cast<uint8_t*>(std::malloc(pixels * sizeof(uint8_t)));
  std::memset(debug_g_buffer_, 0, pixels * sizeof(uint8_t));
  debug_b_buffer_ =
      static_cast<uint8_t*>(std::malloc(pixels * sizeof(uint8_t)));
  std::memset(debug_b_buffer_, 0, pixels * sizeof(uint8_t));

  {
    cudaMallocManaged(reinterpret_cast<void**>(&dsu_buffer_),
                      pixels * sizeof(uint32_t));
    std::memset(dsu_buffer_, 0, pixels * sizeof(uint32_t));
  }

  // Views borrow the fixed-size buffers allocated above.
  segmented_apriltag_r_view_ = {.data = debug_r_buffer_,
                                .stride = width_,
                                .height = height_,
                                .width = width_};
  segmented_apriltag_g_view_ = {.data = debug_g_buffer_,
                                .stride = width_,
                                .height = height_,
                                .width = width_};
  segmented_apriltag_b_view_ = {.data = debug_b_buffer_,
                                .stride = width_,
                                .height = height_,
                                .width = width_};
  max_view_ = {.data = max_buffer_,
               .stride = width_ / 4,
               .height = height_ / 4,
               .width = width_ / 4};
  min_view_ = {.data = min_buffer_,
               .stride = width_ / 4,
               .height = height_ / 4,
               .width = width_ / 4};
  binarized_apriltag_view_ = {.data = binarized_apriltag_buffer_,
                              .stride = width_,
                              .height = height_,
                              .width = width_};
  threshold_view_ = {.data = threshold_buffer_,
                     .stride = width_ / 4,
                     .height = height_ / 4,
                     .width = width_ / 4};
  valid_view_ = {.data = valid_buffer_,
                 .stride = width_ / 4,
                 .height = height_ / 4,
                 .width = width_ / 4};
  segmented_apriltag_view_ = {.data = segmented_apriltag_buffer_,
                              .stride = width_,
                              .height = height_,
                              .width = width_};
  boundary_segmented_apriltag_view_ = {
      .data = boundary_segmented_apriltag_buffer_,
      .stride = width_,
      .height = height_,
      .width = width_};
  sorted_boundary_segmented_apriltag_view_ = {
      .data = sorted_boundary_segmented_apriltag_buffer_,
      .stride = width_,
      .height = height_,
      .width = width_};
  candidate_quad_corners_apriltag_view_ = {
      .data = candidate_quad_corners_apriltag_buffer_,
      .stride = width_,
      .height = height_,
      .width = width_};
  quad_apriltag_view_ = {.data = quad_apriltag_buffer_,
                         .stride = width_,
                         .height = height_,
                         .width = width_};
  bit_locations_apriltag_view_ = {.data = bit_locations_apriltag_buffer_,
                                  .stride = width_,
                                  .height = height_,
                                  .width = width_};
  refined_points_apriltag_view_ = {.data = refined_points_apriltag_buffer_,
                                   .stride = width_,
                                   .height = height_,
                                   .width = width_};
  dsu_view_ = {
      .data = dsu_buffer_,
      .stride = width_,
      .height = height_,
      .width = width_,
  };

  CreateCudaGraph();
}

GpuApriltagDetector::~GpuApriltagDetector() {
  CHECK(cudaGraphExecDestroy(graph_exec_) == cudaSuccess);
  CHECK(cudaGraphDestroy(graph_) == cudaSuccess);
  FreeBuffers();
  CHECK(cudaStreamDestroy(stream_) == cudaSuccess);
}

void GpuApriltagDetector::FreeBuffers() {
  cudaFree(graph_input_buffer_);
  cudaFree(max_buffer_);
  cudaFree(min_buffer_);
  std::free(threshold_buffer_);
  std::free(valid_buffer_);
  cudaFree(binarized_apriltag_buffer_);
  cudaFree(segmented_apriltag_buffer_);
  std::free(boundary_segmented_apriltag_buffer_);
  std::free(sorted_boundary_segmented_apriltag_buffer_);
  std::free(candidate_quad_corners_apriltag_buffer_);
  std::free(quad_apriltag_buffer_);
  std::free(bit_locations_apriltag_buffer_);
  std::free(refined_points_apriltag_buffer_);
  std::free(debug_r_buffer_);
  std::free(debug_g_buffer_);
  std::free(debug_b_buffer_);
  cudaFree(dsu_buffer_);
}

void GpuApriltagDetector::ImWrite(const std::string& path,
                                  const ImageView<uint8_t>& image) {
  cv::Mat mat(image.height, image.width, CV_8UC1, image.data, image.stride);
  cv::imwrite(path, mat);
}

void GpuApriltagDetector::ImWrite(const std::string& path,
                                  const ImageView<uint8_t>& image_r,
                                  const ImageView<uint8_t>& image_g,
                                  const ImageView<uint8_t>& image_b) {
  CHECK_EQ(image_r.width, image_g.width);
  CHECK_EQ(image_r.width, image_b.width);
  CHECK_EQ(image_r.height, image_g.height);
  CHECK_EQ(image_r.height, image_b.height);

  cv::Mat r(image_r.height, image_r.width, CV_8UC1, image_r.data,
            image_r.stride);
  cv::Mat g(image_g.height, image_g.width, CV_8UC1, image_g.data,
            image_g.stride);
  cv::Mat b(image_b.height, image_b.width, CV_8UC1, image_b.data,
            image_b.stride);

  cv::Mat color;
  cv::merge(std::vector<cv::Mat>{b, g, r}, color);

  cv::imwrite(path, color);
}

void GpuApriltagDetector::ImWrite(const std::string& path,
                                  ImageView<uint32_t> segmented_apriltag) {
  const size_t pixels = static_cast<size_t>(width_) * height_;
  std::memset(debug_r_buffer_, 0, pixels * sizeof(uint8_t));
  std::memset(debug_g_buffer_, 0, pixels * sizeof(uint8_t));
  std::memset(debug_b_buffer_, 0, pixels * sizeof(uint8_t));

  for (int i = 0; i < segmented_apriltag.height; i++) {
    for (int j = 0; j < segmented_apriltag.width; j++) {
      uint32_t id = segmented_apriltag(i, j);
      if (id != 0) {
        uint8_t r, g, b;
        PopulateColor(id, r, g, b);
        segmented_apriltag_r_view_(i, j) = r;
        segmented_apriltag_g_view_(i, j) = g;
        segmented_apriltag_b_view_(i, j) = b;
      }
    }
  }

  ImWrite(path, segmented_apriltag_r_view_, segmented_apriltag_g_view_,
          segmented_apriltag_b_view_);
}

void GpuApriltagDetector::PopulateMinMax(ImageView<uint8_t> apriltag,
                                         ImageView<uint8_t> min,
                                         ImageView<uint8_t> max) {
  for (int i = 0; i < min.height; i++) {
    for (int j = 0; j < min.width; j++) {
      GetMinMax(apriltag, i, j, min(i, j), max(i, j));
    }
  }
}

void GpuApriltagDetector::PopulateThresholdValid(ImageView<uint8_t> min,
                                                 ImageView<uint8_t> max,
                                                 ImageView<uint8_t> threshold,
                                                 ImageView<uint8_t> valid) {
  CHECK_EQ(min.width, threshold.width);
  CHECK_EQ(min.height, threshold.height);
  CHECK_EQ(valid.width, threshold.width);
  CHECK_EQ(valid.height, threshold.height);

  for (int i = 0; i < min.height; i++) {
    threshold(i, 0) = (min(i, 0) / 2) + (max(i, 0) / 2);
    threshold(i, min.width - 1) =
        (min(i, min.width - 1) / 2) + (max(i, min.width - 1) / 2);
  }
  for (int j = 0; j < min.width; j++) {
    threshold(0, j) = (min(0, j) / 2) + (max(0, j) / 2);
    threshold(min.height - 1, j) =
        (min(min.height - 1, j) / 2) + (max(min.height - 1, j) / 2);
  }
  for (int i = 1; i < min.height - 1; i++) {
    for (int j = 1; j < min.width - 1; j++) {
      uint8_t max_value = std::max({
          max(i - 1, j - 1),
          max(i - 1, j + 0),
          max(i - 1, j + 1),

          max(i + 0, j - 1),
          max(i + 0, j + 0),
          max(i + 0, j + 1),

          max(i + 1, j - 1),
          max(i + 1, j + 0),
          max(i + 1, j + 1),
      });

      uint8_t min_value = std::min({
          min(i - 1, j - 1),
          min(i - 1, j + 0),
          min(i - 1, j + 1),

          min(i + 0, j - 1),
          min(i + 0, j + 0),
          min(i + 0, j + 1),

          min(i + 1, j - 1),
          min(i + 1, j + 0),
          min(i + 1, j + 1),
      });
      threshold(i, j) = (max_value / 2) + (min_value / 2);
      valid(i, j) = max_value - min_value > 50 ? 255 : 0;
    }
  }
}

void GpuApriltagDetector::PopulateBinarizedApriltag(
    ImageView<uint8_t> threshold, ImageView<uint8_t> valid,
    ImageView<uint8_t> apriltag, ImageView<uint8_t> binarized_apriltag) {
  CHECK_EQ(threshold.height * 4, apriltag.height);
  CHECK_EQ(threshold.width * 4, apriltag.width);
  CHECK_EQ(threshold.height * 4, binarized_apriltag.height);
  CHECK_EQ(threshold.width * 4, binarized_apriltag.width);
  CHECK_EQ(threshold.height, valid.height);
  CHECK_EQ(threshold.width, valid.width);

  for (int j = 0; j < threshold.width; j++) {
    for (int i = 0; i < threshold.height; i++) {
      ApplyThreshold(threshold(i, j), valid(i, j), apriltag, binarized_apriltag,
                     i, j);
    }
  }
}

void GpuApriltagDetector::Segment(int row, int col,
                                  ImageView<uint8_t> binarized_apriltag,
                                  ImageView<uint32_t> segmented_apriltag,
                                  int32_t id) {
  std::queue<Coord<int>> q;
  q.emplace(row, col);
  uint8_t color = binarized_apriltag(row, col);
  segmented_apriltag(row, col) = id;
  while (!q.empty()) {
    Coord<int> coords = q.front();
    q.pop();
    constexpr std::array<int, 4> dx_array = {0, 0, 1, -1};
    constexpr std::array<int, 4> dy_array = {-1, 1, 0, 0};
    for (int i = 0; i < 4; i++) {
      int new_row = coords.row + dx_array[i];
      int new_col = coords.col + dy_array[i];
      if (new_col < 0 || new_row < 0 || new_col >= binarized_apriltag.width ||
          new_row >= binarized_apriltag.height) [[unlikely]] {
        continue;
      }
      if (binarized_apriltag(new_row, new_col) != color) {
        continue;
      }
      if (segmented_apriltag(new_row, new_col) != 0) {
        continue;
      }
      segmented_apriltag(new_row, new_col) = id;
      q.emplace(new_row, new_col);
    }
  }
}

void GpuApriltagDetector::PopulateSegmentedApriltag(
    ImageView<uint8_t> binarized_apriltag,
    ImageView<uint32_t> segmented_apriltag) {
  CHECK_EQ(binarized_apriltag.width, segmented_apriltag.width);
  CHECK_EQ(binarized_apriltag.height, segmented_apriltag.height);
  int id = 1;
  for (int i = 0; i < binarized_apriltag.height; i++) {
    for (int j = 0; j < binarized_apriltag.width; j++) {
      if ((binarized_apriltag(i, j) == 255 || binarized_apriltag(i, j) == 0) &&
          segmented_apriltag(i, j) == 0) {
        Segment(i, j, binarized_apriltag, segmented_apriltag, id++);
      }
    }
  }
}

auto GpuApriltagDetector::GetSegment(ImageView<uint32_t>& segmented_apriltag,
                                     int16_t row, int16_t col)
    -> std::pmr::vector<Coord<int16_t>> {
  constexpr size_t min_segment_size = 128;
  constexpr size_t max_segment_size = 512;
  constexpr int max_revisited = 4;
  ushort num_revisited = 0;
  uint current_direction = 0;
  std::pmr::vector<Coord<int16_t>> segment{&segment_resource_};
  segment.reserve(max_segment_size);
  Coord<int16_t> curr{.row = row, .col = col};
  const Coord<int> start{.row = row, .col = col};
  segment.push_back({.row = row, .col = col});
  const uint32_t id = segmented_apriltag(row, col);
  const uint32_t visited_id = UINT32_MAX - segmented_apriltag(row, col);
  while (segment.size() <= max_segment_size && num_revisited < max_revisited) {
    constexpr std::array<int16_t, 4> drow = {0, 1, 0, -1};
    constexpr std::array<int16_t, 4> dcol = {1, 0, -1, 0};
    bool has_neighbor = false;
    for (int i = 0; i < 4; i++) {
      int new_direction = (current_direction + i) % 4;
      Coord<int16_t> new_coord = {
          .row = static_cast<int16_t>(curr.row + drow[new_direction]),
          .col = static_cast<int16_t>(curr.col + dcol[new_direction])};
      if (new_coord.row >= 0 && new_coord.row < segmented_apriltag.height &&
          new_coord.col >= 0 && new_coord.col < segmented_apriltag.width) {
        if (segmented_apriltag(new_coord.row, new_coord.col) == id ||
            segmented_apriltag(new_coord.row, new_coord.col) == visited_id) {
          num_revisited +=
              segmented_apriltag(new_coord.row, new_coord.col) == visited_id;
          segment.push_back(new_coord);
          segmented_apriltag(new_coord.row, new_coord.col) = UINT32_MAX - id;
          curr = new_coord;
          current_direction = new_direction + 3;
          has_neighbor = true;
          break;
        }
      }
    }
    if (!has_neighbor) {
      return std::pmr::vector<Coord<int16_t>>{&segment_resource_};
    }
    if (start.row == curr.row && start.col == curr.col) {
      return segment.size() > min_segment_size
                 ? segment
                 : std::pmr::vector<Coord<int16_t>>{&segment_resource_};
    }
  }
  return std::pmr::vector<Coord<int16_t>>{&segment_resource_};
}

auto GpuApriltagDetector::GetSegments(ImageView<uint32_t> segmented_apriltag)
    -> std::pmr::vector<std::pmr::vector<Coord<int16_t>>> {
  std::pmr::vector<std::pmr::vector<Coord<int16_t>>> segments{
      &segment_resource_};
  CHECK_LT(static_cast<uint32_t>(segmented_apriltag.width *
                                 segmented_apriltag.height),
           UINT32_MAX - static_cast<uint32_t>(segmented_apriltag.width *
                                              segmented_apriltag.height));
  uint32_t threshold =
      UINT32_MAX - (segmented_apriltag.width * segmented_apriltag.height);
  for (int i = 0; i < segmented_apriltag.height; i++) {
    for (int j = 0; j < segmented_apriltag.width; j++) {
      const uint32_t value = segmented_apriltag(i, j);
      // Valid GPU labels are pixel indices; exclude invalid/visited markers
      // before using a label to index the reusable array.
      if (value < threshold && !visited_segment_ids_[value]) {
        auto segment = GetSegment(segmented_apriltag, i, j);
        if (!segment.empty()) {
          segments.push_back(std::move(segment));
        }
        visited_segment_ids_[value] = 1;
      }
    }
  }
  return segments;
}

void GpuApriltagDetector::PopulateBoundarySegmentedApriltag(
    std::pmr::vector<std::pmr::vector<Coord<int16_t>>>& segments,
    ImageView<uint32_t> boundary_segmented_apriltag) {
  int id = 1;
  for (const auto& pixel_coords : segments) {
    for (const auto& pixel_coord : pixel_coords) {
      boundary_segmented_apriltag(pixel_coord.row, pixel_coord.col) = id;
    }
    id++;
  }
}

auto GpuApriltagDetector::SortSegments(
    std::pmr::vector<std::pmr::vector<Coord<int>>>& segments) {
  for (auto& segment : segments) {
    auto sum = std::accumulate(
        segment.begin(), segment.end(), Coord<int>{.row = 0, .col = 0},
        [](Coord<int> sum, Coord<int> value) -> Coord<int> {
          sum.row += value.row;
          sum.col += value.col;
          return sum;
        });
    Coord<int> mean{.row = sum.row / static_cast<int>(segment.size()),
                    .col = sum.col / static_cast<int>(segment.size())};

    std::ranges::sort(segment, [&mean](Coord<int> a, Coord<int> b) -> bool {
      const int64_t a_row = static_cast<int64_t>(a.row) - mean.row;
      const int64_t a_col = static_cast<int64_t>(a.col) - mean.col;

      const int64_t b_row = static_cast<int64_t>(b.row) - mean.row;
      const int64_t b_col = static_cast<int64_t>(b.col) - mean.col;

      auto sector = [](int64_t x, int64_t y) -> int {
        if (x == 0 && y < 0)
          return 0;  // angle = pi
        if (x > 0)
          return 1;  // (0, pi)
        if (x == 0)
          return 2;  // angle = 0
        return 3;    // (-pi, 0)
      };

      const int sa = sector(a_row, a_col);
      const int sb = sector(b_row, b_col);

      if (sa != sb) {
        return sa < sb;
      }

      // Equivalent angular ordering without atan2.
      return a_col * b_row - a_row * b_col < 0;
    });
  }
}

void GpuApriltagDetector::PopulateSortedBoundarySegmentedApriltag(
    std::pmr::vector<std::pmr::vector<Coord<int16_t>>>& segments,
    ImageView<uint8_t> sorted_boundary_segmented_apriltag) {
  for (auto& segment : segments) {
    float size = segment.size();
    for (size_t i = 0; i < segment.size(); i++) {
      uint8_t value = (i / size) * 255;
      sorted_boundary_segmented_apriltag(segment[i].row, segment[i].col) =
          value;
    }
  }
}

auto GpuApriltagDetector::GetMses(
    std::pmr::vector<std::pmr::vector<Coord<int16_t>>>& segments)
    -> std::vector<std::vector<float>> {
  std::vector<std::vector<float>> mses;
  constexpr int window_size = 50;
  constexpr float window_size_float = window_size;
  for (const auto& segment : segments) {
    Coord<int64_t> first_moment{.row = 0, .col = 0};
    Coord<int64_t> second_moment{.row = 0, .col = 0};
    int64_t xy_moment = 0;
    for (int i = 0; i < window_size; i++) {
      const Coord<int64_t> point{.row = segment[i].row, .col = segment[i].col};
      first_moment.row += point.row;
      first_moment.col += point.col;

      second_moment.row += point.row * point.row;
      second_moment.col += point.col * point.col;

      xy_moment += point.row * point.col;
    }

    int window_head = window_size;
    int window_tail = 0;
    std::vector<float> mse(segment.size());
    for (size_t i = window_size / 2; i < segment.size() + (window_size / 2);
         i++) {
      auto mean_x = first_moment.row / window_size_float;
      auto mean_y = first_moment.col / window_size_float;

      const float cxx = second_moment.row / window_size_float - mean_x * mean_x;
      const float cyy = second_moment.col / window_size_float - mean_y * mean_y;
      const float cxy = (xy_moment / window_size_float) -
                        ((first_moment.row / window_size_float) *
                         (first_moment.col / window_size_float));

      const float a = 1;
      const float b = -(cxx + cyy);
      const float c = (cxx * cyy) - (cxy * cxy);
      auto lambdas = SolveQuadratic(a, b, c);
      if (lambdas.first < lambdas.second) {
        std::swap(lambdas.first, lambdas.second);
      }

      mse[i % mse.size()] = lambdas.second;

      const Coord<int64_t> head{.row = segment[window_head].row,
                                .col = segment[window_head].col};
      first_moment.row += head.row;
      first_moment.col += head.col;
      second_moment.row += head.row * head.row;
      second_moment.col += head.col * head.col;
      xy_moment += head.row * head.col;

      const Coord<int64_t> tail{.row = segment[window_tail].row,
                                .col = segment[window_tail].col};
      first_moment.row -= tail.row;
      first_moment.col -= tail.col;
      second_moment.row -= tail.row * tail.row;
      second_moment.col -= tail.col * tail.col;
      xy_moment -= tail.row * tail.col;

      window_head++;
      window_head %= segment.size();
      window_tail++;
      window_tail %= segment.size();
    }
    mses.push_back(std::move(mse));
  }
  CHECK_EQ(mses.size(), segments.size());
  return mses;
}

auto GpuApriltagDetector::GetCandidatesQuadCorners(
    const std::pmr::vector<std::pmr::vector<Coord<int16_t>>>& segments,
    const std::vector<std::vector<float>>& mse_map)
    -> std::vector<CandidatesQuad> {
  std::vector<CandidatesQuad> quads;
  CHECK_EQ(mse_map.size(), segments.size());
  for (size_t idx = 0; idx < mse_map.size(); idx++) {
    const auto& segment = segments[idx];
    const auto& mse = mse_map[idx];
    CHECK_EQ(segment.size(), mse.size());

    size_t window_size = std::max<size_t>(3, mse.size() / 8);
    if (window_size % 2 == 0) {
      ++window_size;
    }

    CandidatesQuad quad{};
    std::array<float, quad.corners.size()> max_mse{};

    for (size_t i = 0; i < mse.size(); i++) {
      const size_t center = (i + window_size / 2) % mse.size();
      const float middle_mse = mse[center];
      if (middle_mse < max_mse[0]) {
        continue;
      }

      bool peak = true;
      for (size_t j = i; j < i + window_size; j++) {
        if (middle_mse < mse[(j + (window_size / 2)) % mse.size()]) {
          peak = false;
          break;
        }
      }
      if (peak) {
        max_mse[0] = middle_mse;
        quad.corners[0] = segment[(i + (window_size / 2)) % segment.size()];
        for (size_t k = 1; k < max_mse.size(); k++) {
          if (max_mse[k - 1] > max_mse[k]) {
            std::swap(max_mse[k - 1], max_mse[k]);
            std::swap(quad.corners[k - 1], quad.corners[k]);
          }
        }
        i += window_size / 2;
      }
    }
    quads.push_back(quad);
  }
  return quads;
}

void GpuApriltagDetector::PopulateCandidateQuadCornersApriltagBuffer(
    std::vector<CandidatesQuad>& quads,
    ImageView<uint8_t> candidates_quad_corners_apriltag) {
  for (const auto& quad : quads) {
    int color = 255;
    for (const auto& corner : quad.corners) {
      if (corner.row == 0 && corner.col == 0) {
        continue;
      }
      for (int i = -3; i <= 3; i++) {
        for (int j = -3; j <= 3; j++) {
          candidates_quad_corners_apriltag(corner.row + i, corner.col + j) =
              color;
        }
      }
      color -= 50;
    }
  }
}

auto GpuApriltagDetector::GetQuads(
    std::vector<CandidatesQuad>& candidate_quad_corners) -> std::vector<Quad> {
  std::vector<Quad> quads;
  quads.reserve(candidate_quad_corners.size());
  for (const auto& candidate_quad_corner : candidate_quad_corners) {
    constexpr auto candidates = candidate_quad_corner.corners.size();
    Quad quad{
        candidate_quad_corner.corners[candidates - 1],
        candidate_quad_corner.corners[candidates - 2],
        candidate_quad_corner.corners[candidates - 3],
        candidate_quad_corner.corners[candidates - 4],
    };
    quads.push_back(quad);
  }
  return quads;
}

void GpuApriltagDetector::OrderQuads(std::vector<Quad>& quads) {
  for (auto& quad : quads) {
    Coord<int16_t> mean = std::accumulate(
        quad.corners.begin(), quad.corners.end(), Coord<int16_t>{},
        [](Coord<int16_t> sum, Coord<int16_t> value) -> Coord<int16_t> {
          sum.row += value.row;
          sum.col += value.col;
          return sum;
        });
    mean.row /= 4;
    mean.col /= 4;
    std::ranges::sort(
        quad.corners, [&mean](Coord<int16_t> a, Coord<int16_t> b) -> bool {
          const int64_t a_row = static_cast<int64_t>(a.row) - mean.row;
          const int64_t a_col = static_cast<int64_t>(a.col) - mean.col;

          const int64_t b_row = static_cast<int64_t>(b.row) - mean.row;
          const int64_t b_col = static_cast<int64_t>(b.col) - mean.col;

          auto sector = [](int64_t x, int64_t y) -> int {
            if (x == 0 && y < 0)
              return 0;  // angle = pi
            if (x > 0)
              return 1;  // (0, pi)
            if (x == 0)
              return 2;  // angle = 0
            return 3;    // (-pi, 0)
          };

          const int sa = sector(a_row, a_col);
          const int sb = sector(b_row, b_col);

          if (sa != sb) {
            return sa < sb;
          }

          // Equivalent angular ordering without atan2.
          return a_col * b_row - a_row * b_col < 0;
        });
  }
}

void GpuApriltagDetector::PopulateQuadApriltagBuffer(
    std::vector<Quad>& quads, ImageView<uint8_t> quad_apriltag) {
  for (const auto& quad : quads) {
    CHECK(quad.corners.size() == 4);
    int color = 255;
    for (const auto& corner : quad.corners) {
      if (corner.row == 0 && corner.col == 0) {
        continue;
      }
      for (int i = -3; i <= 3; i++) {
        for (int j = -3; j <= 3; j++) {
          quad_apriltag(corner.row + i, corner.col + j) = color;
        }
      }
      color -= 50;
    }
  }
}

auto GpuApriltagDetector::GetBitLocationsHomography(std::vector<Quad>& quads,
                                                    int width, int height)
    -> std::vector<BitLocation> {
  std::vector<BitLocation> bit_locations;
  cv::Mat H;
  for (const auto& quad : quads) {
    if (quad.corners[3].row == 0) {
      bit_locations.push_back({});
      continue;
    }
    const std::array<Coord<int16_t>, 4>& corners = quad.corners;
    BitLocation bit_location;
    {
      // Coordinates are (row, col). Match OrderQuads' winding; swapping
      // these axes reflects the bit grid, which rotations cannot decode.
      std::vector<cv::Point2f> srcPoints{
          {0.5, 0.5}, {8.5, 0.5}, {8.5, 8.5}, {0.5, 8.5}};
      std::vector<cv::Point2f> dstPoints{{static_cast<float>(corners[0].row),
                                          static_cast<float>(corners[0].col)},
                                         {static_cast<float>(corners[1].row),
                                          static_cast<float>(corners[1].col)},
                                         {static_cast<float>(corners[2].row),
                                          static_cast<float>(corners[2].col)},
                                         {static_cast<float>(corners[3].row),
                                          static_cast<float>(corners[3].col)}};
      H = cv::getPerspectiveTransform(srcPoints, dstPoints);
    }
    {
      for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
          std::vector<cv::Point2f> src{
              cv::Point2f(static_cast<float>(i), static_cast<float>(j))};
          std::vector<cv::Point2f> dst;
          cv::perspectiveTransform(src, dst, H);
          bit_location[i][j] = {.row = static_cast<int>(dst[0].x),
                                .col = static_cast<int>(dst[0].y)};
          bit_location[i][j].row =
              std::clamp(bit_location[i][j].row, 0, height - 1);
          bit_location[i][j].col =
              std::clamp(bit_location[i][j].col, 0, width - 1);
        }
      }
    }
    bit_locations.push_back(bit_location);
  }
  return bit_locations;
}

auto GpuApriltagDetector::GetBitLocations(std::vector<Quad>& quads)
    -> std::vector<BitLocation> {
  std::vector<BitLocation> bit_locations;
  for (const auto& quad : quads) {
    if (quad.corners[3].row == 0) {
      bit_locations.push_back({});
      continue;
    }

    std::pair<float, float> first_row_vector{
        quad.corners[1].row - quad.corners[0].row,
        quad.corners[1].col - quad.corners[0].col};
    first_row_vector.first /= 8;
    first_row_vector.second /= 8;

    std::pair<float, float> second_row_vector{
        quad.corners[2].row - quad.corners[3].row,
        quad.corners[2].col - quad.corners[3].col};
    second_row_vector.first /= 8;
    second_row_vector.second /= 8;

    std::pair<float, float> first_col_vector{
        quad.corners[3].row - quad.corners[0].row,
        quad.corners[3].col - quad.corners[0].col};
    first_col_vector.first /= 8;
    first_col_vector.second /= 8;

    std::pair<float, float> second_col_vector{
        quad.corners[2].row - quad.corners[1].row,
        quad.corners[2].col - quad.corners[1].col};
    second_col_vector.first /= 8;
    second_col_vector.second /= 8;

    std::pair<float, float> first_row_offset{quad.corners[0].row,
                                             quad.corners[0].col};
    first_row_offset.first -= first_row_vector.first / 2;
    first_row_offset.second -= first_row_vector.second / 2;

    std::pair<float, float> second_row_offset{quad.corners[3].row,
                                              quad.corners[3].col};
    second_row_offset.first -= second_row_vector.first / 2;
    second_row_offset.second -= second_row_vector.second / 2;

    std::pair<float, float> first_col_offset{quad.corners[0].row,
                                             quad.corners[0].col};
    first_col_offset.first -= first_col_vector.first / 2;
    first_col_offset.second -= first_col_vector.second / 2;

    std::pair<float, float> second_col_offset{quad.corners[1].row,
                                              quad.corners[1].col};
    second_col_offset.first -= second_col_vector.first / 2;
    second_col_offset.second -= second_col_vector.second / 2;

    BitLocation bit_location;
    bool valid = true;
    for (int i = 0; i < 10; i++) {

      std::pair<float, float> first_row_position{
          first_row_offset.first + (i * first_row_vector.first),
          first_row_offset.second + (i * first_row_vector.second)};

      std::pair<float, float> second_row_position{
          second_row_offset.first + (i * second_row_vector.first),
          second_row_offset.second + (i * second_row_vector.second)};

      std::pair<float, float> row_vector{
          second_row_position.first - first_row_position.first,
          second_row_position.second - first_row_position.second};

      for (int j = 0; j < 10; j++) {
        std::pair<float, float> first_col_position{
            first_col_offset.first + (j * first_col_vector.first),
            first_col_offset.second + (j * first_col_vector.second)};
        std::pair<float, float> second_col_position{
            second_col_offset.first + (j * second_col_vector.first),
            second_col_offset.second + (j * second_col_vector.second)};

        float x1 = first_row_position.first;
        float y1 = first_row_position.second;

        float x2 = second_row_position.first;
        float y2 = second_row_position.second;

        float x3 = first_col_position.first;
        float y3 = first_col_position.second;

        float x4 = second_col_position.first;
        float y4 = second_col_position.second;

        float numerator = ((x4 - x3) * (y3 - y1) - (y4 - y3) * (x3 - x1));
        float denomenator = ((x4 - x3) * (y2 - y1) - (y4 - y3) * (x2 - x1));
        float alpha = numerator / denomenator;
        Coord<int> intersection{
            .row = static_cast<int>(first_row_position.first +
                                    row_vector.first * alpha),
            .col = static_cast<int>(first_row_position.second +
                                    row_vector.second * alpha)};
        bit_location[i][j] = intersection;
        if (intersection.row < 0 || intersection.col < 0) {
          valid = false;
        }
      }
    }
    bit_locations.push_back(valid ? bit_location : BitLocation{});
  }
  return bit_locations;
}

void GpuApriltagDetector::PopulateBitLocationsApriltag(
    std::vector<BitLocation>& bit_locations,
    ImageView<uint32_t> bit_locations_apriltag) {
  int idx = 0;
  for (const auto& bit_location : bit_locations) {
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        bit_locations_apriltag(bit_location[i][j].row, bit_location[i][j].col) =
            idx * 2222009;
      }
    }
    idx++;
  }
}

auto GpuApriltagDetector::GetBlackWhiteThreshold(
    ImageView<uint8_t> apriltag, const BitLocation& bit_location)
    -> std::array<float, 3> {
  // Fit intensity = row_slope * (row - 4.5) + col_slope * (col - 4.5)
  // + mean independently on the white and black borders. A single mean
  // threshold misclassifies bits when illumination varies across the tag.
  auto fit_border = [&](int first, int last) -> std::array<float, 3> {
    float row_intensity = 0;
    float col_intensity = 0;
    float row_squared = 0;
    float col_squared = 0;
    float intensity_sum = 0;
    int count = 0;
    for (int row = first; row <= last; ++row) {
      for (int col = first; col <= last; ++col) {
        if (row != first && row != last && col != first && col != last) {
          continue;
        }
        const auto& point = bit_location[row][col];
        const float intensity = apriltag(point.row, point.col);
        const float r = row - 4.5f;
        const float c = col - 4.5f;
        row_intensity += r * intensity;
        col_intensity += c * intensity;
        row_squared += r * r;
        col_squared += c * c;
        intensity_sum += intensity;
        ++count;
      }
    }
    // The centered, symmetric border makes the normal matrix diagonal.
    return std::array<float, 3>{row_intensity / row_squared,
                                col_intensity / col_squared,
                                intensity_sum / count};
  };
  const auto white = fit_border(0, 9);
  const auto black = fit_border(1, 8);
  return {(white[0] + black[0]) / 2, (white[1] + black[1]) / 2,
          (white[2] + black[2]) / 2};
}

auto GpuApriltagDetector::GetTagIds(std::vector<BitLocation>& bit_locations,
                                    ImageView<uint8_t> apriltag)
    -> std::pair<std::vector<int>, std::vector<int>> {
  std::vector<int> tag_ids;
  std::vector<int> rotations;
  tag_ids.reserve(bit_locations.size());
  rotations.reserve(bit_locations.size());
  for (const auto& bit_location : bit_locations) {
    uint64_t code = 0;
    const auto threshold_plane = GetBlackWhiteThreshold(apriltag, bit_location);
    for (uint32_t j = 0; j < family_->nbits; j++) {
      const auto x = family_->bit_x[j];
      const auto y = family_->bit_y[j];

      const float threshold =
          threshold_plane[0] * (static_cast<float>(y) - 3.5f) +
          threshold_plane[1] * (static_cast<float>(x) - 3.5f) +
          threshold_plane[2];
      code <<= 1;
      if (apriltag(bit_location[y + 1][x + 1].row,
                   bit_location[y + 1][x + 1].col) > threshold) {
        code |= 1ULL;
      }
    }
    int tag_id = -1;
    int rotation = -1;
    for (int j = 0; j < 4; j++) {
      constexpr int nbits = 36;
      constexpr int shift = 9;
      constexpr uint64_t mask = (1ULL << nbits) - 1;

      for (uint32_t k = 0; k < family_->ncodes; k++) {
        int hamming = std::popcount(code ^ family_->codes[k]);
        if (hamming <= 2) {
          tag_id = k;
          rotation = j;
          break;
        }
      }
      code = ((code << shift) | (code >> (nbits - shift))) & mask;
    }
    tag_ids.push_back(tag_id);
    rotations.push_back(rotation);
  }
  return {tag_ids, rotations};
}

void GpuApriltagDetector::RotateQuads(std::vector<Quad>& quads,
                                      const std::vector<int>& rotations) {
  CHECK_EQ(quads.size(), rotations.size());
  Quad tmp_quad;
  for (size_t i = 0; i < quads.size(); i++) {
    if (rotations[i] == -1) {
      continue;
    }
    static_assert(tmp_quad.corners.size() == 4);
    for (int j = 0; j < 4; j++) {
      tmp_quad.corners[(j + rotations[i]) % 4] = quads[i].corners[j];
    }
    for (int j = 0; j < 4; j++) {
      quads[i].corners[j] = tmp_quad.corners[j];
    }
  }
}

void GpuApriltagDetector::DrawTagDetections(
    cv::Mat& image, const std::vector<ApriltagDetection>& detections) {

  for (const auto& detection : detections) {
    std::array<cv::Point, 4> points;

    for (int j = 0; j < 4; ++j) {
      points[j] = cv::Point{static_cast<int>(detection.quad.corners[j].col),
                            static_cast<int>(detection.quad.corners[j].row)};
    }

    for (int j = 0; j < 4; ++j) {
      cv::line(image, points[j], points[(j + 1) % 4], cv::Scalar(0, 255, 0), 2);
    }

    cv::circle(image, points[0], 5, cv::Scalar(0, 0, 255), -1);
    cv::putText(image, "ID: " + std::to_string(detection.id),
                points[0] + cv::Point(5, -5), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                cv::Scalar(0, 255, 0), 2);
  }
}

auto GpuApriltagDetector::GradientCol(Coord<int> point,
                                      ImageView<uint8_t>& apriltag) -> float {
  constexpr std::array<std::array<int, 5>, 5> gradient_x{{
      {{-1, -2, 0, 2, 1}},
      {{-4, -8, 0, 8, 4}},
      {{-6, -12, 0, 12, 6}},
      {{-4, -8, 0, 8, 4}},
      {{-1, -2, 0, 2, 1}},
  }};
  float output = 0;
  point.row -= 2;
  point.col -= 2;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      output += gradient_x[i][j] * apriltag(point.row + i, point.col + j);
    }
  }
  return output;
}

auto GpuApriltagDetector::GradientRow(Coord<int> point,
                                      ImageView<uint8_t>& apriltag) -> float {
  constexpr std::array<std::array<int, 5>, 5> gradient_y{{
      {{-1, -4, -6, -4, -1}},
      {{-2, -8, -12, -8, -2}},
      {{0, 0, 0, 0, 0}},
      {{2, 8, 12, 8, 2}},
      {{1, 4, 6, 4, 1}},
  }};
  float output = 0;
  point.row -= 2;
  point.col -= 2;
  for (int i = 0; i < 5; i++) {
    for (int j = 0; j < 5; j++) {
      output += gradient_y[i][j] * apriltag(point.row + i, point.col + j);
    }
  }
  return output;
}

auto GpuApriltagDetector::GetRefinedPoints(
    const std::vector<ApriltagDetection>& apriltag_detections,
    ImageView<uint8_t>& apriltag)
    -> std::vector<std::array<std::vector<WeightedPoint>, 4>> {

  constexpr int num_samples = 10;
  constexpr int search_vector_length = 10;
  constexpr int quad_size = 4;
  std::vector<std::array<std::vector<WeightedPoint>, quad_size>> refined_points;
  for (const auto& apriltag_detection : apriltag_detections) {
    CHECK(apriltag_detection.quad.corners.size() == quad_size);
    const auto& quad = apriltag_detection.quad;
    std::array<std::vector<WeightedPoint>, quad_size> weighted_points;
    for (size_t i = 0; i < quad.corners.size(); i++) {
      weighted_points[i].reserve(num_samples);
      const auto& p1 = quad.corners[i];
      const auto& p2 = quad.corners[(i + 1) % 4];
      const auto& p3 = quad.corners[(i + 2) % 4];
      const std::pair<float, float> v1{
          static_cast<float>(p2.row - p1.row) / num_samples,
          static_cast<float>(p2.col - p1.col) / num_samples};
      const std::pair<float, float> v2{
          static_cast<float>(p3.row - p2.row) / num_samples,
          static_cast<float>(p3.col - p2.col) / num_samples};
      const float v1_cross_v2 = v1.first * v2.second - v1.second * v2.first;
      const int v1_cross_v2_sign = (v1_cross_v2 < 0.0f) ? -1 : 1;

      std::pair<float, float> search_vector{std::abs(v1.second),
                                            std::abs(v1.first)};
      const float magnitude =
          std::hypot(search_vector.first, search_vector.second);
      search_vector.first /= magnitude;
      search_vector.second /= magnitude;
      search_vector.first *= search_vector_length;
      search_vector.second *= search_vector_length;

      for (int j = 0; j < num_samples; j++) {
        const Coord<int> point{.row = static_cast<int>(p1.row + j * v1.first),
                               .col = static_cast<int>(p1.col + j * v1.second)};

        if (point.row - 2 < 0 || point.col - 2 < 0 ||
            point.row + 2 >= apriltag.height || point.col + 2 >= apriltag.width)
            [[unlikely]] {
          continue;
        }

        const std::pair<float, float> gradient{GradientRow(point, apriltag),
                                               GradientCol(point, apriltag)};
        const float v1_cross_gradient =
            v1.first * gradient.second - v1.second * gradient.first;
        if ((v1_cross_gradient < 0) != (v1_cross_v2 < 0)) {
          const Coord<int> start{
              .row = std::max(static_cast<int>(point.row - search_vector.first),
                              2),
              .col = std::max(
                  static_cast<int>(point.col - search_vector.second), 2)};
          const Coord<int> end{
              .row = std::min(static_cast<int>(point.row + search_vector.first),
                              apriltag.height - 3),
              .col =
                  std::min(static_cast<int>(point.col + search_vector.second),
                           apriltag.width - 3)};
          float min_cross = std::numeric_limits<float>::max();
          Coord<int> best_point{.row = -1, .col = -1};
          for (int row = start.row; row <= end.row; row++) {
            for (int col = start.col; col <= end.col; col++) {
              const Coord<int> point{.row = row, .col = col};
              const std::pair<float, float> candidate_gradient{
                  GradientRow(point, apriltag), GradientCol(point, apriltag)};

              const float v1_cross_candidate_gradient =
                  v1.first * candidate_gradient.second -
                  v1.second * candidate_gradient.first;
              if (v1_cross_candidate_gradient * v1_cross_v2_sign < min_cross) {
                min_cross = v1_cross_candidate_gradient * v1_cross_v2_sign;
                best_point = point;
              }
            }
          }
          if (min_cross != std::numeric_limits<float>::max() && min_cross < 0) {
            weighted_points[i].emplace_back(best_point, -min_cross);
          }
        }
      }
    }
    refined_points.push_back(weighted_points);
  }
  return refined_points;
}

void GpuApriltagDetector::PopulateRefinedPointsApriltag(
    const std::vector<std::array<std::vector<WeightedPoint>, 4>>&
        refined_points,
    ImageView<uint8_t>& refined_points_apriltag) {
  constexpr int marker_half_size = 2;
  for (const auto& quad : refined_points) {
    for (const auto& segment : quad) {
      for (const auto& weighted_point : segment) {
        Coord start{
            .row = std::max(weighted_point.coord.row - marker_half_size, 0),
            .col = std::max(weighted_point.coord.col - marker_half_size, 0)};
        Coord end{.row = std::min(weighted_point.coord.row + marker_half_size,
                                  refined_points_apriltag.height - 1),
                  .col = std::min(weighted_point.coord.col + marker_half_size,
                                  refined_points_apriltag.width - 1)};
        for (int row = start.row; row <= end.row; row++) {
          for (int col = start.col; col <= end.col; col++) {
            refined_points_apriltag(row, col) = 255;
          }
        }
      }
    }
  }
}

auto GpuApriltagDetector::Cross(const Coord<float>& a, const Coord<float>& b)
    -> float {
  return a.row * b.col - a.col * b.row;
}

auto GpuApriltagDetector::GetIntersection(
    const Coord<float>& centroid_a, const std::pair<float, float>& vector_a,
    const Coord<float>& centroid_b, const std::pair<float, float>& vector_b)
    -> Coord<int16_t> {
  const float denominator =
      vector_a.first * vector_b.second - vector_a.second * vector_b.first;

  const std::pair<float, float> difference{
      centroid_b.row - centroid_a.row,
      centroid_b.col - centroid_a.col,
  };

  const float t = (difference.first * vector_b.second -
                   difference.second * vector_b.first) /
                  denominator;

  return Coord<int16_t>{
      .row = static_cast<int16_t>(centroid_a.row + t * vector_a.first),
      .col = static_cast<int16_t>(centroid_a.col + t * vector_a.second),
  };
}

auto GpuApriltagDetector::GetRefinedQuads(
    const std::vector<std::array<std::vector<WeightedPoint>, 4>>&
        refined_points) -> std::vector<Quad> {
  std::vector<Quad> refined_quads;
  refined_quads.reserve(refined_points.size());
  std::vector<std::pair<float, float>> vectors;
  std::vector<Coord<float>> centroids;
  vectors.reserve(4);
  centroids.reserve(4);
  for (const auto& tag : refined_points) {
    vectors.clear();
    centroids.clear();
    for (const auto& segment : tag) {
      Coord<float> first_moment{.row = 0, .col = 0};
      Coord<float> second_moment{.row = 0, .col = 0};
      float xy_moment = 0;
      float weight_sum = 0;
      for (const auto& point : segment) {
        first_moment.row += point.coord.row * point.weight;
        first_moment.col += point.coord.col * point.weight;

        second_moment.row += point.coord.row * point.coord.row * point.weight;
        second_moment.col += point.coord.col * point.coord.col * point.weight;

        xy_moment += point.coord.col * point.coord.row * point.weight;

        weight_sum += point.weight;
      }

      float mean_x = first_moment.row / weight_sum;
      float mean_y = first_moment.col / weight_sum;

      const float cxx = second_moment.row / weight_sum - mean_x * mean_x;
      const float cyy = second_moment.col / weight_sum - mean_y * mean_y;
      const float cxy =
          (xy_moment / weight_sum) -
          ((first_moment.row / weight_sum) * (first_moment.col / weight_sum));

      const float angle = 0.5f * std::atan2(2.0f * cxy, cxx - cyy);
      const std::pair<float, float> vector{
          std::cos(angle),
          std::sin(angle),
      };
      const Coord<float> centroid{
          .row = first_moment.row / weight_sum,
          .col = first_moment.col / weight_sum,
      };
      vectors.push_back(vector);
      centroids.push_back(centroid);
    }
    Quad quad;
    for (size_t i = 0; i < quad.corners.size(); i++) {
      quad.corners[(i + 1) % quad.corners.size()] = GetIntersection(
          centroids[i], vectors[i], centroids[(i + 1) % quad.corners.size()],
          vectors[(i + 1) % quad.corners.size()]);
    }
    refined_quads.push_back(quad);
  }
  return refined_quads;
}

void GpuApriltagDetector::ClearBuffers() {
  const size_t pixels = static_cast<size_t>(width_) * height_;
  std::memset(max_buffer_, 0, pixels / 16 * sizeof(uint8_t));
  std::memset(min_buffer_, 0, pixels / 16 * sizeof(uint8_t));
  std::memset(threshold_buffer_, 0, pixels / 16 * sizeof(uint8_t));
  std::memset(valid_buffer_, 0, pixels / 16 * sizeof(uint8_t));
  std::memset(binarized_apriltag_buffer_, 0, pixels * sizeof(uint8_t));
  std::memset(segmented_apriltag_buffer_, 0, pixels * sizeof(uint32_t));
  std::memset(boundary_segmented_apriltag_buffer_, 0,
              pixels * sizeof(uint32_t));
  std::memset(sorted_boundary_segmented_apriltag_buffer_, 0,
              pixels * sizeof(uint8_t));
  std::memset(candidate_quad_corners_apriltag_buffer_, 0,
              pixels * sizeof(uint8_t));
  std::memset(quad_apriltag_buffer_, 0, pixels * sizeof(uint8_t));
  std::memset(bit_locations_apriltag_buffer_, 0, pixels * sizeof(uint32_t));
  std::memset(refined_points_apriltag_buffer_, 0, pixels * sizeof(uint8_t));
  std::ranges::fill(visited_segment_ids_, uint8_t{0});
}

auto GpuApriltagDetector::DetectAprilTag(
    ImageView<uint8_t> apriltag, const std::filesystem::path& output_directory)
    -> std::vector<ApriltagDetection> {
  CHECK(apriltag.data != nullptr);
  CHECK_EQ(apriltag.height, height_);
  CHECK_EQ(apriltag.width, width_);
  CHECK_EQ(apriltag.stride, apriltag.width);
  if (!output_directory.empty()) {
    std::filesystem::create_directories(output_directory);
  }
  ClearBuffers();

  // The graph captures a fixed device-accessible address and packed stride.
  // Refresh its contents from the packed input image for every call.
  std::memcpy(graph_input_buffer_, apriltag.data,
              static_cast<size_t>(width_) * height_ * sizeof(uint8_t));

  // {
  //   PopulateMinMax(apriltag, min_view_, max_view_);
  //   PopulateThresholdValid(min_view_, max_view_, threshold_view_, valid_view_);
  //   PopulateBinarizedApriltag(threshold_view_, valid_view_, apriltag,
  //                            binarized_apriltag_view_);
  //   if (!output_directory.empty()) {
  //     ImWrite((output_directory / "threshold.png").string(), threshold_view_);
  //     ImWrite((output_directory / "valid.png").string(), valid_view_);
  //   }
  // }

  // {
  //   PopulateSegmentedApriltag(binarized_apriltag_view_,
  //                             segmented_apriltag_view_);
  // }
  {
    CHECK(cudaGraphLaunch(graph_exec_, stream_) == cudaSuccess);
    CHECK(cudaStreamSynchronize(stream_) == cudaSuccess);
  }

  if (!output_directory.empty()) {
    ImWrite((output_directory / "max.png").string(), max_view_);
    ImWrite((output_directory / "min.png").string(), min_view_);
    ImWrite((output_directory / "binarized_apriltag.png").string(),
            binarized_apriltag_view_);
    ImWrite((output_directory / "segmented_apriltag.png").string(),
            segmented_apriltag_view_);
    ImWrite((output_directory / "dsu.png").string(), dsu_view_);
  }

  segments_ = GetSegments(dsu_view_);
  // SortSegments(segments_);

  if (!output_directory.empty()) {
    PopulateBoundarySegmentedApriltag(segments_,
                                      boundary_segmented_apriltag_view_);
    ImWrite((output_directory / "boundary_segmented_apriltag.png").string(),
            boundary_segmented_apriltag_view_);
  }

  if (!output_directory.empty()) {
    PopulateSortedBoundarySegmentedApriltag(
        segments_, sorted_boundary_segmented_apriltag_view_);
    ImWrite(
        (output_directory / "sorted_boundary_segmented_apriltag.png").string(),
        sorted_boundary_segmented_apriltag_view_);
  }

  mses_ = GetMses(segments_);
  CHECK_EQ(mses_.size(), segments_.size());

  candidate_quad_corners_ = GetCandidatesQuadCorners(segments_, mses_);
  CHECK_EQ(candidate_quad_corners_.size(), segments_.size());

  if (!output_directory.empty()) {
    memcpy(candidate_quad_corners_apriltag_buffer_,
           sorted_boundary_segmented_apriltag_buffer_,
           sizeof(uint8_t) * apriltag.width * apriltag.height);
    PopulateCandidateQuadCornersApriltagBuffer(
        candidate_quad_corners_, candidate_quad_corners_apriltag_view_);
    ImWrite((output_directory / "candidate_quad_corners_apriltag.png").string(),
            candidate_quad_corners_apriltag_view_);
  }

  memcpy(quad_apriltag_buffer_, sorted_boundary_segmented_apriltag_buffer_,
         sizeof(uint8_t) * apriltag.width * apriltag.height);
  quads_ = GetQuads(candidate_quad_corners_);
  OrderQuads(quads_);
  CHECK_EQ(quads_.size(), segments_.size());
  PopulateQuadApriltagBuffer(quads_, quad_apriltag_view_);
  if (!output_directory.empty()) {
    ImWrite((output_directory / "quad_apriltag.png").string(),
            quad_apriltag_view_);
  }

  bit_locations_ =
      GetBitLocationsHomography(quads_, apriltag.width, apriltag.height);
  // bit_locations_ = GetBitLocations(quads_);

  if (!output_directory.empty()) {
    memcpy(bit_locations_apriltag_buffer_, boundary_segmented_apriltag_buffer_,
           sizeof(uint32_t) * apriltag.width * apriltag.height);
    PopulateBitLocationsApriltag(bit_locations_, bit_locations_apriltag_view_);
    ImWrite((output_directory / "bit_locations_apriltag.png").string(),
            bit_locations_apriltag_view_);
  }

  auto [tag_ids, rotations] = GetTagIds(bit_locations_, apriltag);
  RotateQuads(quads_, rotations);

  detections_.clear();
  CHECK_EQ(tag_ids.size(), rotations.size());
  for (size_t i = 0; i < tag_ids.size(); i++) {
    if (tag_ids[i] != -1) {
      detections_.emplace_back(quads_[i], tag_ids[i]);
    }
  }

  refined_points_ = GetRefinedPoints(detections_, apriltag);

  if (!output_directory.empty()) {
    memcpy(refined_points_apriltag_buffer_,
           sorted_boundary_segmented_apriltag_buffer_,
           sizeof(uint8_t) * apriltag.width * apriltag.height);
    PopulateRefinedPointsApriltag(refined_points_,
                                  refined_points_apriltag_view_);
    ImWrite((output_directory / "refined_points_apriltag.png").string(),
            refined_points_apriltag_view_);
  }

  refined_quads_ = GetRefinedQuads(refined_points_);

  std::vector<ApriltagDetection> refined_detections;
  size_t refined_index = 0;
  for (int& i : tag_ids) {
    if (i != -1) {
      refined_detections.emplace_back(refined_quads_[refined_index], i);
      ++refined_index;
    }
  }

  return refined_detections;
}

void GpuApriltagDetector::CreateCudaGraph() {
  cudaStream_t stream;

  CHECK(cudaStreamCreate(&stream) == cudaSuccess);
  CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal) ==
        cudaSuccess);

  PopulateMinMaxGPU(graph_input_view_, min_view_, max_view_, stream);
  PopulateThresholdValidGPU(graph_input_view_, min_view_, max_view_,
                            binarized_apriltag_view_, stream);
  PopulateSegmentedApriltagGPU(binarized_apriltag_view_,
                               segmented_apriltag_view_, dsu_view_, stream);

  CHECK(cudaStreamEndCapture(stream, &graph_) == cudaSuccess);
  CHECK(cudaStreamDestroy(stream) == cudaSuccess);

  CHECK(cudaGraphInstantiate(&graph_exec_, graph_, 0) == cudaSuccess);
}

}  // namespace apriltag
