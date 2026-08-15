#include <algorithm>
#include <opencv2/opencv.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

#include "control_loop/timer.h"

extern "C" {
#include <apriltag.h>
#include <tag36h11.h>
}

using std::pair;

struct ImageView {
  uint8_t* data;
  uint stride;
  uint height;
  uint width;

  auto operator()(size_t row, size_t col) -> uint8_t& {
    return data[row * stride + col];
  }
};

struct ImageView32 {
  uint32_t* data;
  uint stride;
  uint height;
  uint width;

  auto operator()(size_t row, size_t col) -> uint32_t& {
    return data[row * stride + col];
  }
};

struct Quad {
  std::array<std::pair<int, int>, 4> corners{};
};

struct CandidatesQuad {
  std::array<std::pair<uint, uint>, 4> corners{};
};

struct ApriltagDetection {
  Quad quad;
  int id;
};

using BitLocation = std::array<std::array<std::pair<uint, uint>, 10>, 10>;

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

void PopulateColor(uint id, uint8_t& r, uint8_t& g, uint8_t& b) {
  r = (id * 2222009) % 256;
  g = (id * 2222022) % 256;
  b = (id * 2222222) % 256;
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

  for (uint i = 0; i < segmented_apriltag.height; i++) {
    for (uint j = 0; j < segmented_apriltag.width; j++) {
      uint8_t id = segmented_apriltag(i, j);
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

void GetMinMax(ImageView apriltag, uint row, uint col, uint8_t& min,
               uint8_t& max) {
  row *= 4;
  col *= 4;
  min = apriltag(row, col);
  max = apriltag(row, col);
  for (uint i = 0; i < 4; i++) {
    for (uint j = 0; j < 4; j++) {
      min = std::min(min, (apriltag(row + i, col + j)));
      max = std::max(min, (apriltag(row + i, col + j)));
    }
  }
}

void PopulateMinMax(ImageView apriltag, ImageView min, ImageView max) {
  for (uint i = 0; i < min.height; i++) {
    for (uint j = 0; j < min.width; j++) {
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

  for (uint i = 0; i < min.height; i++) {
    threshold(i, 0) = (min(i, 0) / 2) + (max(i, 0) / 2);
    threshold(i, min.width - 1) =
        (min(i, min.width - 1) / 2) + (max(i, min.width - 1) / 2);
  }
  for (uint j = 0; j < min.width; j++) {
    threshold(0, j) = (min(0, j) / 2) + (max(0, j) / 2);
    threshold(min.height - 1, j) =
        (min(min.height - 1, j) / 2) + (max(min.height - 1, j) / 2);
  }
  for (uint i = 1; i < min.height - 1; i++) {
    for (uint j = 1; j < min.width - 1; j++) {
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

inline void ApplyThreshold(uint8_t threshold, uint8_t valid, ImageView apriltag,
                           ImageView binarized_apriltag, uint row, uint col) {
  row *= 4;
  col *= 4;
  if (valid == 0) {
    for (uint i = row; i < row + 4; i++) {
      for (uint j = col; j < col + 4; j++) {
        binarized_apriltag(i, j) =
            (apriltag(i, j) > threshold) ? (255 / 2) + 50 : (255 / 2) - 50;
      }
    }
  } else {
    for (uint i = row; i < row + 4; i++) {
      for (uint j = col; j < col + 4; j++) {
        binarized_apriltag(i, j) = (apriltag(i, j) > threshold) ? 255 : 0;
      }
    }
  }
}

void PopulateBinarizedApriltag(ImageView threshold, ImageView valid,
                               ImageView apriltag,
                               ImageView binarized_apriltag) {
  CHECK_EQ(threshold.height * 4, apriltag.height);
  CHECK_EQ(threshold.width * 4, apriltag.width);
  CHECK_EQ(threshold.height * 4, binarized_apriltag.height);
  CHECK_EQ(threshold.width * 4, binarized_apriltag.width);
  CHECK_EQ(threshold.height, valid.height);
  CHECK_EQ(threshold.width, valid.width);

  for (uint j = 0; j < threshold.width; j++) {
    for (uint i = 0; i < threshold.height; i++) {
      ApplyThreshold(threshold(i, j), valid(i, j), apriltag, binarized_apriltag,
                     i, j);
    }
  }
}

void Segment(uint row, uint col, ImageView binarized_apriltag,
             ImageView32 segmented_apriltag, int32_t id) {
  std::queue<std::pair<uint, uint>> q;
  q.emplace(row, col);
  uint8_t color = binarized_apriltag(row, col);
  segmented_apriltag(row, col) = id;
  while (!q.empty()) {
    std::pair<uint, uint> coords = q.front();
    q.pop();
    constexpr std::array<int, 4> dx_array = {0, 0, 1, -1};
    constexpr std::array<int, 4> dy_array = {-1, 1, 0, 0};
    for (const auto& dx : dx_array) {
      for (const auto& dy : dy_array) {
        uint new_row = coords.first + dx;  // uint - int: defined behavior?
        uint new_col = coords.second + dy;
        if (new_col >= binarized_apriltag.width ||
            new_row >= binarized_apriltag.height) {
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
}

void PopulateSegmentedApriltag(ImageView binarized_apriltag,
                               ImageView32 segmented_apriltag) {
  CHECK_EQ(binarized_apriltag.width, segmented_apriltag.width);
  CHECK_EQ(binarized_apriltag.height, segmented_apriltag.height);
  uint id = 1;
  for (uint i = 0; i < binarized_apriltag.height; i++) {
    for (uint j = 0; j < binarized_apriltag.width; j++) {
      if ((binarized_apriltag(i, j) == 255 || binarized_apriltag(i, j) == 0) &&
          segmented_apriltag(i, j) == 0) {
        Segment(i, j, binarized_apriltag, segmented_apriltag, id++);
      }
    }
  }
}
auto GetSegments(ImageView32 segmented_apriltag)
    -> std::vector<std::vector<std::pair<uint, uint>>> {
  absl::flat_hash_map<std::pair<uint, uint>,
                      absl::flat_hash_set<std::pair<uint, uint>>>
      segments_set;
  for (uint i = 1; i < segmented_apriltag.height - 1; i++) {
    for (uint j = 1; j < segmented_apriltag.width - 1; j++) {
      if (segmented_apriltag(i, j) != 0) {
        auto id = segmented_apriltag(i, j);
        constexpr std::array<int, 4> dx_array = {0, 0, 1, -1};
        constexpr std::array<int, 4> dy_array = {-1, 1, 0, 0};
        for (const auto& dx : dx_array) {
          for (const auto& dy : dy_array) {
            auto neighbor_id = segmented_apriltag(i + dx, j + dy);
            if (neighbor_id != 0 && neighbor_id != id) {
              segments_set[{std::max(id, neighbor_id),
                            std::min(id, neighbor_id)}]
                  .emplace(i + dx, j + dy);
            }
          }
        }
      }
    }
  }
  std::vector<std::vector<std::pair<uint, uint>>> segments;
  for (const auto& [ids, pixel_coords_set] : segments_set) {
    constexpr size_t min_segment_size = 500;
    if (pixel_coords_set.size() >= min_segment_size) {
      std::vector<std::pair<uint, uint>> pixel_coords_vector(
          pixel_coords_set.begin(), pixel_coords_set.end());
      segments.push_back(std::move(pixel_coords_vector));
    }
  }
  return segments;
}

void PopulateBoundarySegmentedApriltag(
    std::vector<std::vector<std::pair<uint, uint>>>& segments,
    ImageView32 boundary_segmented_apriltag) {
  uint id = 1;
  for (const auto& pixel_coords : segments) {
    for (const auto& pixel_coord : pixel_coords) {
      boundary_segmented_apriltag(pixel_coord.first, pixel_coord.second) = id;
    }
    id++;
  }
}

auto SortSegments(std::vector<std::vector<std::pair<uint, uint>>>& segments) {
  for (auto& segment : segments) {
    auto sum = std::accumulate(
        segment.begin(), segment.end(), std::pair<uint, uint>{0, 0},
        [](std::pair<uint, uint> sum,
           std::pair<uint, uint> value) -> std::pair<uint, uint> {
          sum.first += value.first;
          sum.second += value.second;
          return sum;
        });
    std::pair<uint, uint> mean{sum.first / segment.size(),
                               sum.second / segment.size()};

    std::ranges::sort(
        segment,
        [&mean](std::pair<uint, uint> a, std::pair<uint, uint> b) -> bool {
          const int64_t a_row = static_cast<int64_t>(a.first) - mean.first;
          const int64_t a_col = static_cast<int64_t>(a.second) - mean.second;

          const int64_t b_row = static_cast<int64_t>(b.first) - mean.first;
          const int64_t b_col = static_cast<int64_t>(b.second) - mean.second;

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

void PopulateSortedBoundarySegmentedApriltag(
    std::vector<std::vector<std::pair<uint, uint>>>& segments,
    ImageView sorted_boundary_segmented_apriltag) {
  for (auto& segment : segments) {
    float size = segment.size();
    for (uint i = 0; i < segment.size(); i++) {
      uint8_t value = (i / size) * 255;
      sorted_boundary_segmented_apriltag(segment[i].first, segment[i].second) =
          value;
    }
  }
}

auto SolveQuadratic(float a, float b, float c) -> std::pair<float, float> {
  float determinant = std::sqrt((b * b) - (4 * a * c));
  return {(-b + determinant) / (2 * a), (-b - determinant) / (2 * a)};
}

auto GetMses(std::vector<std::vector<std::pair<uint, uint>>>& segments)
    -> std::vector<std::vector<float>> {
  std::vector<std::vector<float>> mses;
  constexpr uint window_size = 100;
  constexpr float window_size_float = window_size;
  for (const auto& segment : segments) {
    std::pair<uint, uint> first_moment{0, 0};
    std::pair<uint, uint> second_moment{0, 0};
    uint xy_moment = 0;
    for (uint i = 0; i < window_size; i++) {
      first_moment.first += segment[i].first;
      first_moment.second += segment[i].second;

      second_moment.first += segment[i].first * segment[i].first;
      second_moment.second += segment[i].second * segment[i].second;

      xy_moment += segment[i].first * segment[i].second;
    }

    uint window_head = window_size;
    uint window_tail = 0;
    std::vector<float> mse(segment.size());
    for (uint i = window_size / 2; i < segment.size() + (window_size / 2);
         i++) {
      auto mean_x = first_moment.first / window_size_float;
      auto mean_y = first_moment.second / window_size_float;

      const float cxx =
          second_moment.first / window_size_float - mean_x * mean_x;
      const float cyy =
          second_moment.second / window_size_float - mean_y * mean_y;
      const float cxy = (xy_moment / window_size_float) -
                        ((first_moment.first / window_size_float) *
                         (first_moment.second / window_size_float));

      const float a = 1;
      const float b = -(cxx + cyy);
      const float c = (cxx * cyy) - (cxy * cxy);
      auto lambdas = SolveQuadratic(a, b, c);
      if (lambdas.first < lambdas.second) {
        std::swap(lambdas.first, lambdas.second);
      }

      mse[i % mse.size()] = lambdas.second;

      first_moment.first += segment[window_head].first;
      first_moment.second += segment[window_head].second;
      second_moment.first +=
          segment[window_head].first * segment[window_head].first;
      second_moment.second +=
          segment[window_head].second * segment[window_head].second;
      xy_moment += segment[window_head].first * segment[window_head].second;

      first_moment.first -= segment[window_tail].first;
      first_moment.second -= segment[window_tail].second;
      second_moment.first -=
          segment[window_tail].first * segment[window_tail].first;
      second_moment.second -=
          segment[window_tail].second * segment[window_tail].second;
      xy_moment -= segment[window_tail].first * segment[window_tail].second;

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

auto GetCandidatesQuadCorners(
    const std::vector<std::vector<std::pair<uint, uint>>>& segments,
    const std::vector<std::vector<float>>& mse_map)
    -> std::vector<CandidatesQuad> {
  std::vector<CandidatesQuad> quads;
  CHECK_EQ(mse_map.size(), segments.size());
  for (size_t idx = 0; idx < mse_map.size(); idx++) {
    const auto& segment = segments[idx];
    const auto& mse = mse_map[idx];
    CHECK_EQ(segment.size(), mse.size());
    constexpr uint window_size = 100;
    CandidatesQuad quad{};
    std::array<float, quad.corners.size()> max_mse{};
    for (uint i = 0; i < mse.size(); i++) {
      const float middle_mse = mse[(i + (window_size / 2)) % mse.size()];
      if (middle_mse < max_mse[0]) {
        continue;
      }
      bool peak = true;
      for (uint j = i; j < i + window_size; j++) {
        if (middle_mse < mse[(j + (window_size / 2)) % mse.size()]) {
          peak = false;
          break;
        }
      }
      if (peak) {
        max_mse[0] = middle_mse;
        quad.corners[0] = segment[(i + (window_size / 2)) % segment.size()];
        for (uint k = 1; k < max_mse.size(); k++) {
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

void PopulateCandidateQuadCornersApriltagBuffer(
    std::vector<CandidatesQuad>& quads,
    ImageView candidates_quad_corners_apriltag) {
  for (const auto& quad : quads) {
    int color = 255;
    for (const auto& corner : quad.corners) {
      if (corner.first == 0 && corner.second == 0) {
        continue;
      }
      for (int i = -5; i <= 5; i++) {
        for (int j = -5; j <= 5; j++) {
          candidates_quad_corners_apriltag(corner.first + i,
                                           corner.second + j) = color;
        }
      }
      color -= 50;
    }
  }
}

auto GetQuads(std::vector<CandidatesQuad>& candidate_quad_corners)
    -> std::vector<Quad> {
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

void OrderQuads(std::vector<Quad>& quads) {
  for (auto& quad : quads) {
    std::pair<uint, uint> mean = std::accumulate(
        quad.corners.begin(), quad.corners.end(), std::pair<uint, uint>{},
        [](std::pair<uint, uint> sum,
           std::pair<uint, uint> value) -> std::pair<uint, uint> {
          sum.first += value.first;
          sum.second += value.second;
          return sum;
        });
    mean.first /= 4;
    mean.second /= 4;
    std::ranges::sort(
        quad.corners,
        [&mean](std::pair<uint, uint> a, std::pair<uint, uint> b) -> bool {
          const int64_t a_row = static_cast<int64_t>(a.first) - mean.first;
          const int64_t a_col = static_cast<int64_t>(a.second) - mean.second;

          const int64_t b_row = static_cast<int64_t>(b.first) - mean.first;
          const int64_t b_col = static_cast<int64_t>(b.second) - mean.second;

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

void PopulateQuadApriltagBuffer(std::vector<Quad>& quads,
                                ImageView quad_apriltag) {
  for (const auto& quad : quads) {
    CHECK(quad.corners.size() == 4);
    int color = 255;
    for (const auto& corner : quad.corners) {
      if (corner.first == 0 && corner.second == 0) {
        continue;
      }
      for (int i = -5; i <= 5; i++) {
        for (int j = -5; j <= 5; j++) {
          quad_apriltag(corner.first + i, corner.second + j) = color;
        }
      }
      color -= 50;
    }
  }
}

auto GetBitLocations(std::vector<Quad>& quads) -> std::vector<BitLocation> {
  std::vector<BitLocation> bit_locations;
  for (const auto& quad : quads) {
    if (quad.corners[3].first == 0) {
      bit_locations.push_back({});
      continue;
    }

    std::pair<float, float> first_row_vector{
        quad.corners[1].first - quad.corners[0].first,
        quad.corners[1].second - quad.corners[0].second};
    first_row_vector.first /= 8;
    first_row_vector.second /= 8;

    std::pair<float, float> second_row_vector{
        quad.corners[2].first - quad.corners[3].first,
        quad.corners[2].second - quad.corners[3].second};
    second_row_vector.first /= 8;
    second_row_vector.second /= 8;

    std::pair<float, float> first_col_vector{
        quad.corners[3].first - quad.corners[0].first,
        quad.corners[3].second - quad.corners[0].second};
    first_col_vector.first /= 8;
    first_col_vector.second /= 8;

    std::pair<float, float> second_col_vector{
        quad.corners[2].first - quad.corners[1].first,
        quad.corners[2].second - quad.corners[1].second};
    second_col_vector.first /= 8;
    second_col_vector.second /= 8;

    std::pair<float, float> first_row_offset{quad.corners[0].first,
                                             quad.corners[0].second};
    first_row_offset.first -= first_row_vector.first / 2;
    first_row_offset.second -= first_row_vector.second / 2;

    std::pair<float, float> second_row_offset{quad.corners[3].first,
                                              quad.corners[3].second};
    second_row_offset.first -= second_row_vector.first / 2;
    second_row_offset.second -= second_row_vector.second / 2;

    std::pair<float, float> first_col_offset{quad.corners[0].first,
                                             quad.corners[0].second};
    first_col_offset.first -= first_col_vector.first / 2;
    first_col_offset.second -= first_col_vector.second / 2;

    std::pair<float, float> second_col_offset{quad.corners[1].first,
                                              quad.corners[1].second};
    second_col_offset.first -= second_col_vector.first / 2;
    second_col_offset.second -= second_col_vector.second / 2;

    BitLocation bit_location;
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
        std::pair<int, int> intersection{
            first_row_position.first + row_vector.first * alpha,
            first_row_position.second + row_vector.second * alpha};
        bit_location[i][j] = intersection;
      }
    }
    bit_locations.push_back(bit_location);
  }
  return bit_locations;
}

void PopulateBitLocationsApriltag(std::vector<BitLocation>& bit_locations,
                                  ImageView32 bit_locations_apriltag) {
  int idx = 0;
  for (const auto& bit_location : bit_locations) {
    for (uint i = 0; i < 10; i++) {
      for (uint j = 0; j < 10; j++) {
        bit_locations_apriltag(bit_location[i][j].first,
                               bit_location[i][j].second) = idx * 2222009;
      }
    }
    idx++;
  }
}

void PrintCode(const BitLocation& bit_location, ImageView binarized_apriltag) {
  for (int i = 1; i <= 8; i++) {
    for (int j = 1; j <= 8; j++) {
      std::cout << static_cast<int>(
                       binarized_apriltag(bit_location[i][j].first,
                                          bit_location[i][j].second) > 255 / 2)
                << " ";
    }
    std::cout << "\n";
  }
  std::cout << "----------------------------\n";
}

auto GetBlackWhiteThreshold(ImageView apriltag,
                            const BitLocation& bit_location) {
  float white = 0;
  for (int i = 0; i < 10; i++) {
    white += apriltag(bit_location[0][i].first, bit_location[0][i].second);
    white += apriltag(bit_location[9][i].first, bit_location[9][i].second);
    white += apriltag(bit_location[i][0].first, bit_location[i][0].second);
    white += apriltag(bit_location[i][9].first, bit_location[i][9].second);
  }
  white /= 40;

  float black = 0;
  for (int i = 1; i < 9; i++) {
    black += apriltag(bit_location[1][i].first, bit_location[1][i].second);
    black += apriltag(bit_location[8][i].first, bit_location[8][i].second);
    black += apriltag(bit_location[i][1].first, bit_location[i][1].second);
    black += apriltag(bit_location[i][8].first, bit_location[i][8].second);
  }
  black /= 40;

  return (white + black) / 2;
}
auto GetTagIds(std::vector<BitLocation>& bit_locations, ImageView apriltag,
               apriltag_family_t* family)
    -> std::pair<std::vector<int>, std::vector<int>> {
  std::vector<int> tag_ids;
  std::vector<int> rotations;
  tag_ids.reserve(bit_locations.size());
  rotations.reserve(bit_locations.size());
  for (const auto& bit_location : bit_locations) {
    uint64_t code = 0;
    auto threshold = GetBlackWhiteThreshold(apriltag, bit_location);
    for (uint32_t j = 0; j < family->nbits; j++) {
      const auto x = family->bit_x[j];
      const auto y = family->bit_y[j];

      code <<= 1;
      if (apriltag(bit_location[y + 1][x + 1].first,
                   bit_location[y + 1][x + 1].second) > threshold) {
        code |= 1ULL;
      }
    }
    int tag_id = -1;
    int rotation = -1;
    for (int j = 0; j < 4; j++) {
      constexpr int nbits = 36;
      constexpr int shift = 9;
      constexpr uint64_t mask = (1ULL << nbits) - 1;

      for (uint32_t k = 0; k < family->ncodes; k++) {
        int hamming = std::popcount(code ^ family->codes[k]);
        if (hamming <= 2) {
          tag_id = k;
          LOG(INFO) << "Found tag: " << k << " " << hamming << " " << j;
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
      points[j] = cv::Point{static_cast<int>(detection.quad.corners[j].second),
                            static_cast<int>(detection.quad.corners[j].first)};
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

auto DetectAprilTag(ImageView apriltag) -> std::vector<ApriltagDetection> {
  CHECK(apriltag.height % 4 == 0);
  CHECK(apriltag.width % 4 == 0);
  auto* max_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height / 16, sizeof(uint8_t)));
  auto* min_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height / 16, sizeof(uint8_t)));
  ImageView max{.data = max_buffer,
                .stride = apriltag.stride / 4,
                .height = apriltag.height / 4,
                .width = apriltag.width / 4};
  ImageView min{.data = min_buffer,
                .stride = apriltag.stride / 4,
                .height = apriltag.height / 4,
                .width = apriltag.width / 4};
  PopulateMinMax(apriltag, min, max);
  ImWrite("/root/max.png", max);
  ImWrite("/root/min.png", min);

  auto* threshold_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height / 16, sizeof(uint8_t)));
  ImageView threshold{.data = threshold_buffer,
                      .stride = apriltag.stride / 4,
                      .height = apriltag.height / 4,
                      .width = apriltag.width / 4};

  auto* valid_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height / 16, sizeof(uint8_t)));
  ImageView valid{.data = valid_buffer,
                  .stride = apriltag.stride / 4,
                  .height = apriltag.height / 4,
                  .width = apriltag.width / 4};
  PopulateThresholdValid(min, max, threshold, valid);
  ImWrite("/root/threshold.png", threshold);
  ImWrite("/root/valid.png", valid);

  auto* binarized_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
  ImageView binarized_apriltag{.data = binarized_apriltag_buffer,
                               .stride = apriltag.stride,
                               .height = apriltag.height,
                               .width = apriltag.width};

  PopulateBinarizedApriltag(threshold, valid, apriltag, binarized_apriltag);
  ImWrite("/root/binarized_apriltag.png", binarized_apriltag);

  auto* segmented_apriltag_buffer = static_cast<uint32_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint32_t)));
  ImageView32 segmented_apriltag{.data = segmented_apriltag_buffer,
                                 .stride = apriltag.stride,
                                 .height = apriltag.height,
                                 .width = apriltag.width};
  PopulateSegmentedApriltag(binarized_apriltag, segmented_apriltag);
  ImWrite("/root/segmented_apriltag.png", segmented_apriltag);

  auto segments = GetSegments(segmented_apriltag);
  SortSegments(segments);

  auto* boundary_segmented_apriltag_buffer = static_cast<uint32_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint32_t)));
  ImageView32 boundary_segmented_apriltag{
      .data = boundary_segmented_apriltag_buffer,
      .stride = apriltag.stride,
      .height = apriltag.height,
      .width = apriltag.width};
  PopulateBoundarySegmentedApriltag(segments, boundary_segmented_apriltag);
  ImWrite("/root/boundary_segmented_apriltag.png", boundary_segmented_apriltag);

  auto* sorted_boundary_segmented_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
  ImageView sorted_boundary_segmented_apriltag{
      .data = sorted_boundary_segmented_apriltag_buffer,
      .stride = apriltag.stride,
      .height = apriltag.height,
      .width = apriltag.width};

  PopulateSortedBoundarySegmentedApriltag(segments,
                                          sorted_boundary_segmented_apriltag);
  ImWrite("/root/sorted_boundary_segmented_apriltag.png",
          sorted_boundary_segmented_apriltag);

  auto mses = GetMses(segments);
  CHECK_EQ(mses.size(), segments.size());

  auto candidate_quad_corners = GetCandidatesQuadCorners(segments, mses);
  CHECK_EQ(candidate_quad_corners.size(), segments.size());

  auto* candidate_quad_corners_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
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

  auto* quad_apriltag_buffer = static_cast<uint8_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
  memcpy(quad_apriltag_buffer, sorted_boundary_segmented_apriltag_buffer,
         sizeof(uint8_t) * apriltag.width * apriltag.height);
  ImageView quad_apriltag{.data = quad_apriltag_buffer,
                          .stride = apriltag.stride,
                          .height = apriltag.height,
                          .width = apriltag.width};
  auto quads = GetQuads(candidate_quad_corners);
  OrderQuads(quads);
  CHECK_EQ(quads.size(), segments.size());
  PopulateQuadApriltagBuffer(quads, quad_apriltag);
  ImWrite("/root/quad_apriltag.png", quad_apriltag);

  auto bit_locations = GetBitLocations(quads);

  auto* bit_locations_apriltag_buffer = static_cast<uint32_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint32_t)));
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

  apriltag_family_t* family = tag36h11_create();
  auto [tag_ids, rotations] = GetTagIds(bit_locations, apriltag, family);
  RotateQuads(quads, rotations);

  std::vector<ApriltagDetection> detections;
  CHECK_EQ(tag_ids.size(), rotations.size());
  for (size_t i = 0; i < tag_ids.size(); i++) {
    if (tag_ids[i] != -1) {
      detections.emplace_back(quads[i], tag_ids[i]);
    }
  }

  free(max_buffer);
  free(min_buffer);
  free(threshold_buffer);
  free(valid_buffer);
  free(binarized_apriltag_buffer);
  free(segmented_apriltag_buffer);
  free(boundary_segmented_apriltag_buffer);
  free(sorted_boundary_segmented_apriltag_buffer);
  free(candidate_quad_corners_apriltag_buffer);
  free(quad_apriltag_buffer);
  free(bit_locations_apriltag_buffer);

  return detections;
}

auto main() -> int {
  std::string apriltag_path = "/root/apriltag.png";
  cv::Mat apriltag = cv::imread(apriltag_path, cv::IMREAD_GRAYSCALE);
  uint8_t* pixels = apriltag.data;
  uint height = apriltag.rows;
  uint width = apriltag.cols;
  CHECK(apriltag.step == static_cast<size_t>(apriltag.cols));
  constexpr int runs = 25;
  double average_run_time = 0.0;
  for (int i = 0; i < 25; i++) {
    control_loop::Timer timer;
    auto detections = DetectAprilTag(ImageView{
        .data = pixels, .stride = width, .height = height, .width = width});
    average_run_time += timer.Stop().count();
  }
  auto detections = DetectAprilTag(ImageView{
      .data = pixels, .stride = width, .height = height, .width = width});
  auto annotated_apriltag = apriltag.clone();
  DrawTagDetections(annotated_apriltag, detections);
  cv::imwrite("/root/annotated_apriltag.png", annotated_apriltag);
  LOG(INFO) << average_run_time / runs;
}
