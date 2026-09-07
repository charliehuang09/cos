#include "apriltag/gpu_apriltag_detector_lib.h"
#include <tag36h11.h>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/cleanup/cleanup.h"
#include "control_loop/thread_pool.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <future>
#include <span>
#include <limits>
#include <numeric>
#include <opencv2/opencv.hpp>
#include <queue>
#include <ranges>
#include <tuple>

#include <cuda_runtime_api.h>

#define CUDA_CHECK(call)                                         \
  do {                                                           \
    const cudaError_t cuda_check_error = (call);                 \
    if (cuda_check_error != cudaSuccess) {                       \
      std::cerr << cudaGetErrorString(cuda_check_error) << '\n'; \
      std::exit(EXIT_FAILURE);                                   \
    }                                                            \
  } while (0)

namespace {
class ScopedHostRegistration {
 public:
  ScopedHostRegistration(void* data, size_t size) : data_(data) {
    const cudaError_t error =
        cudaHostRegister(data_, size, cudaHostRegisterMapped);
    if (error == cudaSuccess) {
      owns_registration_ = true;
    } else if (error == cudaErrorHostMemoryAlreadyRegistered) {
      // cudaHostRegister also records this otherwise acceptable result as the
      // thread's last CUDA error. Clear it so a later launch check does not
      // report this registration attempt as a kernel failure.
      cudaGetLastError();
    } else {
      CUDA_CHECK(error);
    }
  }

  ScopedHostRegistration(const ScopedHostRegistration&) = delete;
  auto operator=(const ScopedHostRegistration&)
      -> ScopedHostRegistration& = delete;

  ~ScopedHostRegistration() {
    if (!owns_registration_) {
      return;
    }
    const cudaError_t error = cudaHostUnregister(data_);
    if (error != cudaSuccess) {
      std::cerr << cudaGetErrorString(error) << '\n';
    }
  }

 private:
  void* data_;
  bool owns_registration_ = false;
};

[[gnu::always_inline]]
void inline PopulateColor(int id, uint8_t& r, uint8_t& g, uint8_t& b) {
  r = (id * 2222009) % 256;
  g = (id * 2222022) % 256;
  b = (id * 2222222) % 256;
}

[[gnu::always_inline]]
auto inline SolveQuadratic(float a, float b, float c)
    -> std::pair<float, float> {
  float determinant = std::sqrt((b * b) - (4 * a * c));
  return {(-b + determinant) / (2 * a), (-b - determinant) / (2 * a)};
}

[[gnu::always_inline]]
void inline GetMinMax(apriltag::ImageView apriltag, int row, int col,
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
                           apriltag::ImageView apriltag,
                           apriltag::ImageView binarized_apriltag, int row,
                           int col) {
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
               apriltag::ImageView binarized_apriltag) {
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

void OrderQuad(apriltag::Quad& quad) {
  apriltag::Coord<int> mean = std::accumulate(
      quad.corners.begin(), quad.corners.end(), apriltag::Coord<int>{},
      [](apriltag::Coord<int> sum,
         apriltag::Coord<int> value) -> apriltag::Coord<int> {
        sum.row += value.row;
        sum.col += value.col;
        return sum;
      });
  mean.row /= 4;
  mean.col /= 4;
  std::ranges::sort(
      quad.corners,
      [&mean](apriltag::Coord<int> a, apriltag::Coord<int> b) -> bool {
        const int64_t a_row = static_cast<int64_t>(a.row) - mean.row;
        const int64_t a_col = static_cast<int64_t>(a.col) - mean.col;

        const int64_t b_row = static_cast<int64_t>(b.row) - mean.row;
        const int64_t b_col = static_cast<int64_t>(b.col) - mean.col;

        auto sector = [](int64_t x, int64_t y) -> int {
          if (x == 0 && y < 0)
            return 0;  // angle = pi
          if (x > 0)
            return 1;  // (0, pi) if (x == 0) return 2;  // angle = 0
          return 3;    // (-pi, 0)
        };

        const int sa = sector(a_row, a_col);
        const int sb = sector(b_row, b_col);

        if (sa != sb) {
          return sa < sb;
        }

        // Equivalent angular ordering without atan2.
        const int64_t cross = a_col * b_row - a_row * b_col;
      if (cross != 0) return cross < 0;
      return a.row != b.row ? a.row < b.row : a.col < b.col;
      });
}

}  // namespace

namespace apriltag {

void ImWrite(const std::string& path, const ImageView& image) {
  cv::Mat mat(image.height, image.width, CV_8UC1, image.data, image.stride);
  cv::imwrite(path, mat);
}

void ImWrite(const std::string& path, const ImageView& image_r,
             const ImageView& image_g, const ImageView& image_b) {
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

void ImWrite(const std::string& path, ImageView32 segmented_apriltag) {
  auto* segmented_apriltag_buffer_r = static_cast<uint8_t*>(calloc(
      segmented_apriltag.width * segmented_apriltag.height, sizeof(uint8_t)));
  ImageView segmented_apriltag_r{.data = segmented_apriltag_buffer_r,
                                 .stride = segmented_apriltag.stride,
                                 .height = segmented_apriltag.height,
                                 .width = segmented_apriltag.width};
  auto* segmented_apriltag_buffer_g = static_cast<uint8_t*>(calloc(
      segmented_apriltag.width * segmented_apriltag.height, sizeof(uint8_t)));
  ImageView segmented_apriltag_g{.data = segmented_apriltag_buffer_g,
                                 .stride = segmented_apriltag.stride,
                                 .height = segmented_apriltag.height,
                                 .width = segmented_apriltag.width};
  auto* segmented_apriltag_buffer_b = static_cast<uint8_t*>(calloc(
      segmented_apriltag.width * segmented_apriltag.height, sizeof(uint8_t)));
  ImageView segmented_apriltag_b{.data = segmented_apriltag_buffer_b,
                                 .stride = segmented_apriltag.stride,
                                 .height = segmented_apriltag.height,
                                 .width = segmented_apriltag.width};

  for (int i = 0; i < segmented_apriltag.height; i++) {
    for (int j = 0; j < segmented_apriltag.width; j++) {
      int8_t id = segmented_apriltag(i, j);
      if (id != 0) {
        uint8_t r, g, b;
        PopulateColor(id, r, g, b);
        segmented_apriltag_r(i, j) = r;
        segmented_apriltag_g(i, j) = g;
        segmented_apriltag_b(i, j) = b;
      }
    }
  }

  ImWrite(path, segmented_apriltag_r, segmented_apriltag_g,
          segmented_apriltag_b);

  free(segmented_apriltag_buffer_r);
  free(segmented_apriltag_buffer_g);
  free(segmented_apriltag_buffer_b);
}

void PopulateMinMax(ImageView apriltag, ImageView min, ImageView max) {
  for (int i = 0; i < min.height; i++) {
    for (int j = 0; j < min.width; j++) {
      GetMinMax(apriltag, i, j, min(i, j), max(i, j));
    }
  }
}

void PopulateThresholdValid(ImageView min, ImageView max, ImageView threshold,
                            ImageView valid) {
  CHECK_EQ(min.width, threshold.width);
  CHECK_EQ(min.height, threshold.height);
  CHECK_EQ(valid.width, threshold.width);
  CHECK_EQ(valid.height, threshold.height);

  for (int i = 0; i < min.height; i++) {
    for (int j = 0; j < min.width; j++) {
      uint8_t min_value = 255;
      uint8_t max_value = 0;
      for (int di = -1; di <= 1; ++di) {
        int r = std::clamp(i + di, 0, min.height - 1);
        for (int dj = -1; dj <= 1; ++dj) {
          int c = std::clamp(j + dj, 0, min.width - 1);
          min_value = std::min(min_value, min(r, c));
          max_value = std::max(max_value, max(r, c));
        }
      }
      threshold(i, j) = (max_value / 2) + (min_value / 2);
      valid(i, j) = (max_value - min_value > 8) ? 255 : 0;
    }
  }
}

// TODO GPU
void PopulateBinarizedApriltag(ImageView threshold, ImageView valid,
                               ImageView apriltag,
                               ImageView binarized_apriltag) {
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

void Segment(int row, int col, ImageView binarized_apriltag,
             ImageView32 segmented_apriltag, int32_t id) {
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

void PopulateSegmentedApriltag(ImageView binarized_apriltag,
                               ImageView32 segmented_apriltag) {
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

auto GetSegmentsCPU(ImageView32 segmented_apriltag)
    -> std::vector<std::vector<Coord<int>>> {
  absl::flat_hash_map<uint64_t, std::vector<Coord<int>>> segments_set;
  segments_set.reserve(1024);
  const int h = segmented_apriltag.height - 1;
  const int w = segmented_apriltag.width - 1;
  for (int i = 0; i < h; i++) {
    for (int j = 0; j < w; j++) {
      const auto id = segmented_apriltag(i, j);
      if (id == 0) {
        continue;
      }
      const auto right_id = segmented_apriltag(i, j + 1);
      if (right_id != 0 && right_id != id) {
        const uint64_t key =
            (static_cast<uint64_t>(std::max(id, right_id)) << 32) |
            std::min(id, right_id);
        auto& set = segments_set[key];
        set.emplace_back(i, j + 1);
        set.emplace_back(i, j);
      }
      const auto bottom_id = segmented_apriltag(i + 1, j);
      if (bottom_id != 0 && bottom_id != id) {
        const uint64_t key =
            (static_cast<uint64_t>(std::max(id, bottom_id)) << 32) |
            std::min(id, bottom_id);
        auto& set = segments_set[key];
        set.emplace_back(i + 1, j);
        set.emplace_back(i, j);
      }
    }
  }
  std::vector<std::vector<Coord<int>>> segments;
  for (auto& [ids, pixel_coords_vector] : segments_set) {
    constexpr size_t min_segment_size = 80;
    if (pixel_coords_vector.size() >= min_segment_size) {
      segments.push_back(std::move(pixel_coords_vector));
    }
  }
  return segments;
}

void PopulateBoundarySegmentedApriltag(
    std::vector<std::vector<Coord<int>>>& segments,
    ImageView32 boundary_segmented_apriltag) {
  int id = 1;
  for (const auto& pixel_coords : segments) {
    for (const auto& pixel_coord : pixel_coords) {
      boundary_segmented_apriltag(pixel_coord.row, pixel_coord.col) = id;
    }
    id++;
  }
}

void SortSegmentsCPU(std::vector<std::vector<Coord<int>>>& segments) {
  for (auto& segment : segments) {
    if (segment.empty()) {
      continue;
    }
    auto sum = std::accumulate(
        segment.begin(), segment.end(), Coord<int64_t>{.row = 0, .col = 0},
        [](Coord<int64_t> sum, Coord<int> value) -> Coord<int64_t> {
          sum.row += value.row;
          sum.col += value.col;
          return sum;
        });
    Coord<int64_t> mean{.row = sum.row / static_cast<int64_t>(segment.size()),
                        .col = sum.col / static_cast<int64_t>(segment.size())};

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
      const int64_t cross = a_col * b_row - a_row * b_col;
      if (cross != 0) return cross < 0;
      return a.row != b.row ? a.row < b.row : a.col < b.col;
    });

    auto removed = std::ranges::unique(segment);
    segment.erase(removed.begin(), removed.end());
  }
}

void SortSegments(std::vector<std::vector<Coord<int>>>& segments) {
  SortSegmentsGPU(segments);
}

void PopulateSortedBoundarySegmentedApriltag(
    std::vector<std::vector<Coord<int>>>& segments,
    ImageView sorted_boundary_segmented_apriltag) {
  for (auto& segment : segments) {
    float size = segment.size();
    for (size_t i = 0; i < segment.size(); i++) {
      uint8_t value = (i / size) * 255;
      sorted_boundary_segmented_apriltag(segment[i].row, segment[i].col) =
          value;
    }
  }
}

static auto GetMsesForSegments(std::span<const std::vector<Coord<int>>> segments)
    -> std::vector<std::vector<float>> {
  std::vector<std::vector<float>> mses;
  mses.reserve(segments.size());
  for (const auto& segment : segments) {
    if (segment.size() < 4) {
      mses.emplace_back(segment.size(), 0.0f);
      continue;
    }
    const int window_size =
        std::max(4, std::min(40, static_cast<int>(segment.size() / 6)));
    const auto window_size_float = static_cast<float>(window_size);

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

    int window_head = window_size == static_cast<int>(segment.size()) ? 0 : window_size;
    int window_tail = 0;
    std::vector<float> mse(segment.size(), 0.0f);
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

      mse[i < mse.size() ? i : i - mse.size()] = lambdas.second;

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
      if (window_head == static_cast<int>(segment.size())) window_head = 0;
      window_tail++;
      if (window_tail == static_cast<int>(segment.size())) window_tail = 0;
    }
    mses.push_back(std::move(mse));
  }
  CHECK_EQ(mses.size(), segments.size());
  return mses;
}

auto GetMses(std::vector<std::vector<Coord<int>>>& segments)
    -> std::vector<std::vector<float>> {
  return GetMsesForSegments(segments);
}

auto IsApproximateSquare(const Quad& quad) -> bool;

static auto GetCandidatesQuadCornersForSegments(
    std::span<const std::vector<Coord<int>>> segments,
    std::span<const std::vector<float>> mse_map)
    -> std::vector<CandidatesQuad> {
  std::vector<CandidatesQuad> quads;
  CHECK_EQ(mse_map.size(), segments.size());
  quads.reserve(segments.size());

  struct Peak {
    size_t index;
    float val;
  };
  thread_local std::vector<Peak> peaks;
  thread_local std::vector<size_t> chosen_peak_indices;

  for (size_t idx = 0; idx < mse_map.size(); idx++) {
    const auto& segment = segments[idx];
    const auto& mse = mse_map[idx];
    CHECK_EQ(segment.size(), mse.size());

    CandidatesQuad quad{};
    if (segment.size() < 4 || mse.empty()) {
      quads.push_back(quad);
      continue;
    }

    const int window_size =
        std::max(4, std::min(40, static_cast<int>(segment.size() / 6)));
    const size_t peak_radius = std::max(
        size_t{2},
        std::min(static_cast<size_t>(window_size / 2), mse.size() / 8));

    peaks.clear();
    const size_t mse_sz = mse.size();
    for (size_t i = 0; i < mse_sz; i++) {
      const float candidate_mse = mse[i];
      if (!std::isfinite(candidate_mse) || candidate_mse <= 0.001f) {
        continue;
      }

      const size_t prev1_idx = (i > 0) ? (i - 1) : (mse_sz - 1);
      const size_t next1_idx = (i + 1 < mse_sz) ? (i + 1) : 0;
      const float prev1 = mse[prev1_idx];
      const float next1 = mse[next1_idx];
      if (candidate_mse < prev1 || candidate_mse <= next1) {
        continue;
      }

      bool is_peak = true;
      for (size_t offset = 2; offset <= peak_radius; offset++) {
        const size_t prev_idx =
            (i >= offset) ? (i - offset) : (i + mse_sz - offset);
        const size_t next_idx =
            (i + offset < mse_sz) ? (i + offset) : (i + offset - mse_sz);
        const float prev_mse = mse[prev_idx];
        const float next_mse = mse[next_idx];
        if (prev_mse > candidate_mse || next_mse >= candidate_mse) {
          is_peak = false;
          break;
        }
      }

      if (is_peak) {
        peaks.push_back({.index = i, .val = candidate_mse});
      }
    }

    if (peaks.size() > 16) {
      std::partial_sort(peaks.begin(), peaks.begin() + 16, peaks.end(),
                        [](const Peak& a, const Peak& b) {
                          return a.val > b.val;
                        });
      peaks.resize(16);
    } else {
      std::ranges::sort(peaks, [](const Peak& a, const Peak& b) -> bool {
        return a.val > b.val;
      });
    }

    const size_t min_dist = std::max(size_t{2}, mse_sz / 16);
    chosen_peak_indices.clear();
    for (const auto& p : peaks) {
      bool too_close = false;
      for (size_t chosen : chosen_peak_indices) {
        size_t d =
            (p.index >= chosen) ? (p.index - chosen) : (chosen - p.index);
        d = std::min(d, mse_sz - d);
        if (d < min_dist) {
          too_close = true;
          break;
        }
      }
      if (!too_close) {
        chosen_peak_indices.push_back(p.index);
        if (chosen_peak_indices.size() >= 10) {
          break;
        }
      }
    }

    if (chosen_peak_indices.size() < 4) {
      quads.push_back(quad);
      continue;
    }

    std::ranges::sort(chosen_peak_indices);
    const size_t K = chosen_peak_indices.size();

    std::array<Coord<int>, 10> pts{};
    for (size_t i = 0; i < K; ++i) {
      pts[i] = segment[chosen_peak_indices[i]];
    }

    std::array<std::array<float, 10>, 10> len{};
    std::array<std::array<float, 10>, 10> vr{};
    std::array<std::array<float, 10>, 10> vc{};
    for (size_t a = 0; a < K; ++a) {
      for (size_t b = a + 1; b < K; ++b) {
        const float r = static_cast<float>(pts[b].row - pts[a].row);
        const float c = static_cast<float>(pts[b].col - pts[a].col);
        const float l = std::hypot(r, c);
        vr[a][b] = r;
        vr[b][a] = -r;
        vc[a][b] = c;
        vc[b][a] = -c;
        len[a][b] = l;
        len[b][a] = l;
      }
    }

    float best_score = -1.0f;
    std::array<Coord<int>, 4> best_corners{};
    constexpr float cos_threshold = 0.90f;

    for (size_t m0 = 0; m0 < K - 3; ++m0) {
      for (size_t m1 = m0 + 1; m1 < K - 2; ++m1) {
        const float len01 = len[m0][m1];
        if (len01 < 3.0f) {
          continue;
        }
        const float v01_r = vr[m0][m1];
        const float v01_c = vc[m0][m1];

        for (size_t m2 = m1 + 1; m2 < K - 1; ++m2) {
          const float len12 = len[m1][m2];
          if (len12 < 3.0f) {
            continue;
          }
          const float v12_r = vr[m1][m2];
          const float v12_c = vc[m1][m2];

          const float cross01_12 = v01_r * v12_c - v01_c * v12_r;
          if (std::abs(cross01_12) < 1.0f) {
            continue;
          }
          const float cos1 =
              (v01_r * v12_r + v01_c * v12_c) / (len01 * len12);
          if (!std::isfinite(cos1) || std::abs(cos1) > cos_threshold) {
            continue;
          }

          for (size_t m3 = m2 + 1; m3 < K; ++m3) {
            const float len23 = len[m2][m3];
            if (len23 < 3.0f) {
              continue;
            }
            const float v23_r = vr[m2][m3];
            const float v23_c = vc[m2][m3];

            const float cross12_23 = v12_r * v23_c - v12_c * v23_r;
            if ((cross12_23 > 0.0f) != (cross01_12 > 0.0f)) {
              continue;
            }
            const float cos2 =
                (v12_r * v23_r + v12_c * v23_c) / (len12 * len23);
            if (!std::isfinite(cos2) || std::abs(cos2) > cos_threshold) {
              continue;
            }

            const float len30 = len[m3][m0];
            if (len30 < 3.0f) {
              continue;
            }
            const float v30_r = vr[m3][m0];
            const float v30_c = vc[m3][m0];

            const float cross23_30 = v23_r * v30_c - v23_c * v30_r;
            if ((cross23_30 > 0.0f) != (cross01_12 > 0.0f)) {
              continue;
            }
            const float cos3 =
                (v23_r * v30_r + v23_c * v30_c) / (len23 * len30);
            if (!std::isfinite(cos3) || std::abs(cos3) > cos_threshold) {
              continue;
            }

            const float cross30_01 = v30_r * v01_c - v30_c * v01_r;
            if ((cross30_01 > 0.0f) != (cross01_12 > 0.0f)) {
              continue;
            }
            const float cos0 =
                (v30_r * v01_r + v30_c * v01_c) / (len30 * len01);
            if (!std::isfinite(cos0) || std::abs(cos0) > cos_threshold) {
              continue;
            }

            const auto& c0 = pts[m0];
            const auto& c1 = pts[m1];
            const auto& c2 = pts[m2];
            const auto& c3 = pts[m3];

            const float area = 0.5f * std::abs(static_cast<float>(
                                          (c0.col * c1.row - c1.col * c0.row) +
                                          (c1.col * c2.row - c2.col * c1.row) +
                                          (c2.col * c3.row - c3.col * c2.row) +
                                          (c3.col * c0.row - c0.col * c3.row)));
            if (area >= 16.0f) {
              const float mse_sum = mse[chosen_peak_indices[m0]] +
                                    mse[chosen_peak_indices[m1]] +
                                    mse[chosen_peak_indices[m2]] +
                                    mse[chosen_peak_indices[m3]];
              const float score = area * mse_sum;
              if (score > best_score) {
                best_score = score;
                best_corners = {c0, c1, c2, c3};
              }
            }
          }
        }
      }
    }

    if (best_score > 0.0f) {
      Quad q{best_corners};
      OrderQuad(q);
      quad.corners = q.corners;
    }
    quads.push_back(quad);
  }
  return quads;
}

auto GetCandidatesQuadCorners(
    const std::vector<std::vector<Coord<int>>>& segments,
    const std::vector<std::vector<float>>& mse_map)
    -> std::vector<CandidatesQuad> {
  return GetCandidatesQuadCornersForSegments(segments, mse_map);
}

auto GetCandidatesQuadCornersParallel(
    const std::vector<std::vector<Coord<int>>>& segments)
    -> std::vector<CandidatesQuad> {
  auto calculate = [](std::span<const std::vector<Coord<int>>> part) {
    const auto mses = GetMsesForSegments(part);
    return GetCandidatesQuadCornersForSegments(part, mses);
  };
  static const size_t workers = std::min<size_t>(
      2, std::max(1u, std::thread::hardware_concurrency()) - 1);
  if (segments.size() < 64 || workers == 0) return calculate(segments);

  // Shared across detectors: at most two background workers, plus the caller.
  static control_loop::ThreadPool pool(workers);
  std::array<std::future<std::vector<CandidatesQuad>>, 2> futures;
  // Workers reference the caller's segments; finish them even on exceptions.
  absl::Cleanup wait_for_workers = [&] {
    for (auto& future : futures) if (future.valid()) future.wait();
  };
  const std::span<const std::vector<Coord<int>>> all(segments);
  const size_t parts = workers + 1;
  for (size_t i = 0; i < workers; ++i) {
    const size_t first = i * segments.size() / parts;
    const size_t last = (i + 1) * segments.size() / parts;
    auto task = std::make_shared<std::packaged_task<std::vector<CandidatesQuad>()>>(
        [calculate, part = all.subspan(first, last - first)] { return calculate(part); });
    futures[i] = task->get_future();
    pool.Submit([task] { (*task)(); });
  }
  auto last = calculate(all.subspan(workers * segments.size() / parts));
  std::vector<CandidatesQuad> candidates;
  candidates.reserve(segments.size());
  for (size_t i = 0; i < workers; ++i) {
    auto part = futures[i].get();
    candidates.insert(candidates.end(), part.begin(), part.end());
  }
  candidates.insert(candidates.end(), last.begin(), last.end());
  return candidates;
}

void PopulateCandidateQuadCornersApriltagBuffer(
    std::vector<CandidatesQuad>& quads,
    ImageView candidates_quad_corners_apriltag) {
  for (const auto& quad : quads) {
    int color = 255;
    for (const auto& corner : quad.corners) {
      if (corner.row == 0 && corner.col == 0) {
        continue;
      }
      for (int i = -2; i <= 2; i++) {
        const int r = corner.row + i;
        if (r < 0 || r >= candidates_quad_corners_apriltag.height) {
          continue;
        }
        for (int j = -2; j <= 2; j++) {
          const int c = corner.col + j;
          if (c < 0 || c >= candidates_quad_corners_apriltag.width) {
            continue;
          }
          candidates_quad_corners_apriltag(r, c) = color;
        }
      }
      color -= 50;
    }
  }
}

auto IsApproximateSquare(const Quad& quad) -> bool {  // TODO
  static_assert(std::tuple_size_v<decltype(Quad::corners)> == 4);
  constexpr float threshold = 0.90f;
  float initial_cross = 0.0f;
  for (size_t i = 0; i < quad.corners.size(); i++) {
    const auto& c1 = quad.corners[i];
    const auto& c2 = quad.corners[(i + 1) % quad.corners.size()];
    const auto& c3 = quad.corners[(i + 2) % quad.corners.size()];
    std::pair<float, float> v1{static_cast<float>(c2.row - c1.row),
                               static_cast<float>(c2.col - c1.col)};
    std::pair<float, float> v2{static_cast<float>(c3.row - c2.row),
                               static_cast<float>(c3.col - c2.col)};
    float len1 = std::hypot(v1.first, v1.second);
    float len2 = std::hypot(v2.first, v2.second);
    if (len1 < 3.0f || len2 < 3.0f) {
      return false;
    }
    float cross = v1.first * v2.second - v1.second * v2.first;
    if (i == 0) {
      initial_cross = cross;
      if (std::abs(initial_cross) < 1.0f)
        return false;
    } else {
      if ((cross > 0.0f) != (initial_cross > 0.0f))
        return false;
    }
    float cos_theta =
        (v1.first * v2.first + v1.second * v2.second) / (len1 * len2);
    if (!std::isfinite(cos_theta) || std::abs(cos_theta) > threshold) {
      return false;
    }
  }
  return true;
}

auto GetQuads(std::vector<CandidatesQuad>& candidate_quad_corners)
    -> std::vector<Quad> {
  std::vector<Quad> quads;
  quads.reserve(candidate_quad_corners.size());
  for (const auto& candidate_quad_corner : candidate_quad_corners) {
    if (candidate_quad_corner.corners[0].row == 0 &&
        candidate_quad_corner.corners[0].col == 0 &&
        candidate_quad_corner.corners[1].row == 0 &&
        candidate_quad_corner.corners[1].col == 0) {
      continue;
    }
    Quad quad{
        candidate_quad_corner.corners[0],
        candidate_quad_corner.corners[1],
        candidate_quad_corner.corners[2],
        candidate_quad_corner.corners[3],
    };
    OrderQuad(quad);
    if (IsApproximateSquare(quad)) {
      quads.push_back(quad);
    }
  }
  return quads;
}

void PopulateQuadApriltagBuffer(std::vector<Quad>& quads,
                                ImageView quad_apriltag) {
  for (const auto& quad : quads) {
    CHECK(quad.corners.size() == 4);
    int color = 255;
    for (const auto& corner : quad.corners) {
      if (corner.row == 0 && corner.col == 0) {
        continue;
      }
      for (int i = -5; i <= 5; i++) {
        const int r = corner.row + i;
        if (r < 0 || r >= quad_apriltag.height) {
          continue;
        }
        for (int j = -5; j <= 5; j++) {
          const int c = corner.col + j;
          if (c < 0 || c >= quad_apriltag.width) {
            continue;
          }
          quad_apriltag(r, c) = color;
        }
      }
      color -= 50;
    }
  }
}

auto GetBitLocations(std::vector<Quad>& quads) -> std::vector<BitLocation> {
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
        float alpha =
            (std::abs(denomenator) > 1e-6f) ? (numerator / denomenator) : 0.5f;
        Coord<int> intersection{
            .row = static_cast<int>(std::lround(first_row_position.first +
                                                row_vector.first * alpha)),
            .col = static_cast<int>(std::lround(first_row_position.second +
                                                row_vector.second * alpha))};
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

void PopulateBitLocationsApriltag(std::vector<BitLocation>& bit_locations,
                                  ImageView32 bit_locations_apriltag) {
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

auto GetBlackWhiteThreshold(ImageView apriltag,
                            const BitLocation& bit_location) {
  float white = 0;
  int white_count = 0;
  auto sample_white = [&](int r, int c) -> void {
    if (r >= 0 && r < apriltag.height && c >= 0 && c < apriltag.width) {
      white += apriltag(r, c);
      white_count++;
    }
  };
  for (int i = 0; i < 10; i++) {
    sample_white(bit_location[0][i].row, bit_location[0][i].col);
    sample_white(bit_location[9][i].row, bit_location[9][i].col);
    sample_white(bit_location[i][0].row, bit_location[i][0].col);
    sample_white(bit_location[i][9].row, bit_location[i][9].col);
  }
  white = white_count > 0 ? (white / white_count) : 255.0f;

  float black = 0;
  int black_count = 0;
  auto sample_black = [&](int r, int c) -> void {
    if (0 <= r && r < apriltag.height && c >= 0 && c < apriltag.width) {
      black += apriltag(r, c);
      black_count++;
    }
  };
  for (int i = 1; i < 9; i++) {
    sample_black(bit_location[1][i].row, bit_location[1][i].col);
    sample_black(bit_location[8][i].row, bit_location[8][i].col);
    sample_black(bit_location[i][1].row, bit_location[i][1].col);
    sample_black(bit_location[i][8].row, bit_location[i][8].col);
  }
  black = black_count > 0 ? (black / black_count) : 0.0f;

  return (white + black) / 2.0f;
}

auto GetTagIdsCPU(std::vector<BitLocation>& bit_locations, ImageView apriltag,
                  apriltag_family_t* family,
                  const std::vector<int>& target_tag_ids)
    -> std::pair<std::vector<int>, std::vector<int>> {
  std::vector<int> tag_ids;
  std::vector<int> rotations;
  tag_ids.reserve(bit_locations.size());
  rotations.reserve(bit_locations.size());

  std::vector<int> valid_target_ids;
  if (!target_tag_ids.empty()) {
    valid_target_ids.reserve(target_tag_ids.size());
    for (int id : target_tag_ids) {
      if (id >= 0 && static_cast<uint32_t>(id) < family->ncodes) {
        valid_target_ids.push_back(id);
      }
    }
  }

  for (const auto& bit_location : bit_locations) {
    const auto threshold = GetBlackWhiteThreshold(apriltag, bit_location);
    int tag_id = -1;
    int rotation = -1;
    int best_hamming = 3;

    auto try_decode = [&](float thresh) -> void {
      uint64_t code = 0;
      for (uint32_t j = 0; j < family->nbits; j++) {
        const auto x = family->bit_x[j];
        const auto y = family->bit_y[j];

        code <<= 1;
        int r = bit_location[y + 1][x + 1].row;
        int c = bit_location[y + 1][x + 1].col;
        if (r >= 0 && r < apriltag.height && c >= 0 && c < apriltag.width) {
          if (apriltag(r, c) > thresh) {
            code |= 1ULL;
          }
        }
      }

      for (int j = 0; j < 4; j++) {
        constexpr int nbits = 36;
        constexpr int shift = 9;
        constexpr uint64_t mask = (1ULL << nbits) - 1;

        if (!target_tag_ids.empty()) {
          for (int k : valid_target_ids) {
            int hamming = std::popcount(code ^ family->codes[k]);
            if (hamming < best_hamming) {
              best_hamming = hamming;
              tag_id = k;
              rotation = j;
              if (hamming == 0) {
                return;
              }
            }
          }
        } else {
          for (uint32_t k = 0; k < family->ncodes; k++) {
            int hamming = std::popcount(code ^ family->codes[k]);
            if (hamming < best_hamming) {
              best_hamming = hamming;
              tag_id = k;
              rotation = j;
              if (hamming == 0) {
                return;
              }
            }
          }
        }
        if (best_hamming == 0) {
          return;
        }
        code = ((code << shift) | (code >> (nbits - shift))) & mask;
      }
    };

    try_decode(threshold);
    if (tag_id == -1) {
      for (float delta : {-8.0f, 8.0f, -16.0f, 16.0f}) {
        try_decode(threshold + delta);
        if (tag_id != -1) {
          break;
        }
      }
    }

    tag_ids.push_back(tag_id);
    rotations.push_back(rotation);
  }
  return {tag_ids, rotations};
}

auto GetTagIds(std::vector<BitLocation>& bit_locations, ImageView apriltag,
               apriltag_family_t* family,
               const std::vector<int>& target_tag_ids)
    -> std::pair<std::vector<int>, std::vector<int>> {
  if (apriltag.data_gpu != nullptr) {
    return GetTagIdsGPU(bit_locations, apriltag, family, target_tag_ids);
  }
  return GetTagIdsCPU(bit_locations, apriltag, family, target_tag_ids);
}

void RotateQuads(std::vector<Quad>& quads, std::vector<int>& rotations) {
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

void DrawTagDetections(cv::Mat& image,
                       const std::vector<ApriltagDetection>& detections) {

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

auto GradientCol(Coord<int> point, ImageView& apriltag) -> float {
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

auto GradientRow(Coord<int> point, ImageView& apriltag) -> float {
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

auto GetRefinedPoints(const std::vector<ApriltagDetection>& apriltag_detections,
                      ImageView& apriltag)
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

void PopulateRefinedPointsApriltag(
    const std::vector<std::array<std::vector<WeightedPoint>, 4>>&
        refined_points,
    ImageView& refined_points_apriltag) {
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

auto Cross(const Coord<float>& a, const Coord<float>& b) -> float {
  return a.row * b.col - a.col * b.row;
}

auto GetIntersection(const Coord<float>& centroid_a,
                     const std::pair<float, float>& vector_a,
                     const Coord<float>& centroid_b,
                     const std::pair<float, float>& vector_b) -> Coord<int> {
  const float denominator =
      vector_a.first * vector_b.second - vector_a.second * vector_b.first;

  const std::pair<float, float> difference{
      centroid_b.row - centroid_a.row,
      centroid_b.col - centroid_a.col,
  };

  const float t = (difference.first * vector_b.second -
                   difference.second * vector_b.first) /
                  denominator;

  return Coord<int>{
      .row = static_cast<int>(centroid_a.row + t * vector_a.first),
      .col = static_cast<int>(centroid_a.col + t * vector_a.second),
  };
}

auto GetRefinedQuads(
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

[[deprecated]]
auto DetectAprilTag(ImageView apriltag, bool imwrite,
                    const std::vector<int>& target_tag_ids)
    -> std::vector<ApriltagDetection> {
  CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceMapHost));

  const ScopedHostRegistration apriltag_registration(
      apriltag.data, static_cast<size_t>(apriltag.stride) * apriltag.height);
  apriltag.EnableGpu();

  CHECK(apriltag.height % 4 == 0);
  CHECK(apriltag.width % 4 == 0);
  uint8_t* max_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&max_buffer),
                           apriltag.width * apriltag.height / 16,
                           cudaHostAllocMapped));
  uint8_t* min_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&min_buffer),
                           apriltag.width * apriltag.height / 16,
                           cudaHostAllocMapped));
  ImageView max{.data = max_buffer,
                .stride = apriltag.stride / 4,
                .height = apriltag.height / 4,
                .width = apriltag.width / 4};
  ImageView min{.data = min_buffer,
                .stride = apriltag.stride / 4,
                .height = apriltag.height / 4,
                .width = apriltag.width / 4};
  max.EnableGpu();
  min.EnableGpu();

  PopulateMinMaxGPU(apriltag, min, max);
  if (imwrite) {
    ImWrite("/root/max.png", max);
    ImWrite("/root/min.png", min);
  }

  uint8_t* threshold_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&threshold_buffer),
                           apriltag.width * apriltag.height / 16,
                           cudaHostAllocMapped));
  ImageView threshold{.data = threshold_buffer,
                      .stride = apriltag.stride / 4,
                      .height = apriltag.height / 4,
                      .width = apriltag.width / 4};

  uint8_t* valid_buffer;
  CUDA_CHECK(cudaHostAlloc(reinterpret_cast<void**>(&valid_buffer),
                           apriltag.width * apriltag.height / 16,
                           cudaHostAllocMapped));
  ImageView valid{.data = valid_buffer,
                  .stride = apriltag.stride / 4,
                  .height = apriltag.height / 4,
                  .width = apriltag.width / 4};
  threshold.EnableGpu();
  valid.EnableGpu();
  PopulateThresholdValidGPU(min, max, threshold, valid);
  if (imwrite) {
    ImWrite("/root/threshold.png", threshold);
    ImWrite("/root/valid.png", valid);
  }

  auto* binarized_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
  ImageView binarized_apriltag{.data = binarized_apriltag_buffer,
                               .stride = apriltag.stride,
                               .height = apriltag.height,
                               .width = apriltag.width};

  PopulateBinarizedApriltag(threshold, valid, apriltag, binarized_apriltag);
  if (imwrite) {
    ImWrite("/root/binarized_apriltag.png", binarized_apriltag);
  }

  auto* segmented_apriltag_buffer = static_cast<uint32_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint32_t)));
  ImageView32 segmented_apriltag{.data = segmented_apriltag_buffer,
                                 .stride = apriltag.stride,
                                 .height = apriltag.height,
                                 .width = apriltag.width};
  uint8_t* device_image = nullptr;
  uint32_t* device_labels = nullptr;
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_image),
                        apriltag.width * apriltag.height * sizeof(uint8_t)));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&device_labels),
                        apriltag.width * apriltag.height * sizeof(uint32_t)));
  PopulateSegmentedApriltagGPU(binarized_apriltag, segmented_apriltag,
                               device_image, device_labels);
  if (imwrite) {
    ImWrite("/root/segmented_apriltag.png", segmented_apriltag);
  }

  GpuSegmentExtractor segment_extractor;
  auto segments = segment_extractor.ExtractDevice(
      device_labels, apriltag.width, apriltag.height, apriltag.width);
  SortSegments(segments);

  auto* boundary_segmented_apriltag_buffer = static_cast<uint32_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint32_t)));
  ImageView32 boundary_segmented_apriltag{
      .data = boundary_segmented_apriltag_buffer,
      .stride = apriltag.stride,
      .height = apriltag.height,
      .width = apriltag.width};
  PopulateBoundarySegmentedApriltag(segments, boundary_segmented_apriltag);
  if (imwrite) {
    ImWrite("/root/boundary_segmented_apriltag.png",
            boundary_segmented_apriltag);
  }

  auto* sorted_boundary_segmented_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
  ImageView sorted_boundary_segmented_apriltag{
      .data = sorted_boundary_segmented_apriltag_buffer,
      .stride = apriltag.stride,
      .height = apriltag.height,
      .width = apriltag.width};

  PopulateSortedBoundarySegmentedApriltag(segments,
                                          sorted_boundary_segmented_apriltag);
  if (imwrite) {
    ImWrite("/root/sorted_boundary_segmented_apriltag.png",
            sorted_boundary_segmented_apriltag);
  }

  auto mses = GetMses(segments);
  CHECK_EQ(mses.size(), segments.size());

  auto candidate_quad_corners = GetCandidatesQuadCorners(segments, mses);
  CHECK_EQ(candidate_quad_corners.size(), segments.size());

  auto* candidate_quad_corners_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
  if (imwrite) {
    memcpy(candidate_quad_corners_apriltag_buffer,
           sorted_boundary_segmented_apriltag_buffer,
           sizeof(uint8_t) * apriltag.width * apriltag.height);
    ImageView candidate_quad_corners_apriltag{
        .data = candidate_quad_corners_apriltag_buffer,
        .stride = apriltag.stride,
        .height = apriltag.height,
        .width = apriltag.width};
    PopulateCandidateQuadCornersApriltagBuffer(candidate_quad_corners,
                                               candidate_quad_corners_apriltag);
    ImWrite("/root/candidate_quad_corners_apriltag.png",
            candidate_quad_corners_apriltag);
  }

  auto quads = GetQuads(candidate_quad_corners);

  auto* quad_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
  if (imwrite) {
    memcpy(quad_apriltag_buffer, sorted_boundary_segmented_apriltag_buffer,
           sizeof(uint8_t) * apriltag.width * apriltag.height);
    ImageView quad_apriltag{.data = quad_apriltag_buffer,
                            .stride = apriltag.stride,
                            .height = apriltag.height,
                            .width = apriltag.width};

    PopulateQuadApriltagBuffer(quads, quad_apriltag);
    ImWrite("/root/quad_apriltag.png", quad_apriltag);
  }

  auto bit_locations = GetBitLocations(quads);

  auto* bit_locations_apriltag_buffer = static_cast<uint32_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint32_t)));
  if (imwrite) {
    memcpy(bit_locations_apriltag_buffer, boundary_segmented_apriltag_buffer,
           sizeof(uint32_t) * apriltag.width * apriltag.height);
    ImageView32 bit_locations_apriltag{
        .data = bit_locations_apriltag_buffer,
        .stride = apriltag.stride,
        .height = apriltag.height,
        .width = apriltag.width,
    };
    PopulateBitLocationsApriltag(bit_locations, bit_locations_apriltag);
    ImWrite("/root/bit_locations_apriltag.png", bit_locations_apriltag);
  }

  apriltag_family_t* family = tag36h11_create();
  auto [tag_ids, rotations] =
      GetTagIds(bit_locations, apriltag, family, target_tag_ids);
  RotateQuads(quads, rotations);

  std::vector<ApriltagDetection> detections;
  CHECK_EQ(tag_ids.size(), rotations.size());
  for (size_t i = 0; i < tag_ids.size(); i++) {
    if (tag_ids[i] != -1) {
      detections.emplace_back(quads[i], tag_ids[i]);
    }
  }

  auto refined_points = GetRefinedPoints(detections, apriltag);

  auto* refined_points_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
  if (imwrite) {
    ImageView refined_points_apriltag{.data = refined_points_apriltag_buffer,
                                      .stride = apriltag.stride,
                                      .height = apriltag.height,
                                      .width = apriltag.width};
    memcpy(refined_points_apriltag_buffer,
           sorted_boundary_segmented_apriltag_buffer,
           sizeof(uint8_t) * apriltag.width * apriltag.height);

    PopulateRefinedPointsApriltag(refined_points, refined_points_apriltag);
    ImWrite("/root/refined_points_apriltag.png", refined_points_apriltag);
  }

  auto refined_quads = GetRefinedQuads(refined_points);

  std::vector<ApriltagDetection> refined_detections;
  size_t refined_index = 0;
  for (int& i : tag_ids) {
    if (i != -1) {
      refined_detections.emplace_back(refined_quads[refined_index], i);
      ++refined_index;
    }
  }

  cudaFree(device_image);
  cudaFree(device_labels);
  cudaFreeHost(max_buffer);
  cudaFreeHost(min_buffer);
  cudaFreeHost(threshold_buffer);
  cudaFreeHost(valid_buffer);
  free(binarized_apriltag_buffer);
  free(segmented_apriltag_buffer);
  free(boundary_segmented_apriltag_buffer);
  free(sorted_boundary_segmented_apriltag_buffer);
  free(candidate_quad_corners_apriltag_buffer);
  free(quad_apriltag_buffer);
  free(bit_locations_apriltag_buffer);
  free(refined_points_apriltag_buffer);

  return refined_detections;
}

}  // namespace apriltag
