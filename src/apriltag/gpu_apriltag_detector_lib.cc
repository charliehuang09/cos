#include "apriltag/gpu_apriltag_detector_lib.h"
#include <tag36h11.h>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

#include <opencv2/opencv.hpp>

namespace {
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
      max = std::max(min, (apriltag(row + i, col + j)));
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

auto GetSegments(ImageView32 segmented_apriltag)
    -> std::vector<std::vector<Coord<int>>> {
  absl::flat_hash_map<std::pair<uint32_t, uint32_t>,
                      absl::flat_hash_set<Coord<int>>>
      segments_set;
  for (int i = 0; i < segmented_apriltag.height - 1; i += 1) {
    for (int j = 0; j < segmented_apriltag.width - 1; j += 1) {
      if (segmented_apriltag(i, j) != 0) {
        constexpr int dx = 0;
        constexpr int dy = 1;
        auto id = segmented_apriltag(i, j);
        auto neighbor_id = segmented_apriltag(i + dx, j + dy);
        if (neighbor_id != 0 && neighbor_id != id) {
          segments_set[{std::max(id, neighbor_id), std::min(id, neighbor_id)}]
              .emplace(i + dx, j + dy);
          segments_set[{std::max(id, neighbor_id), std::min(id, neighbor_id)}]
              .emplace(i, j);
        }
      }
    }
  }
  for (int i = 0; i < segmented_apriltag.height - 1; i += 1) {
    for (int j = 0; j < segmented_apriltag.width - 1; j += 1) {
      if (segmented_apriltag(i, j) != 0) {
        constexpr int dx = 1;
        constexpr int dy = 0;
        auto id = segmented_apriltag(i, j);
        auto neighbor_id = segmented_apriltag(i + dx, j + dy);
        if (neighbor_id != 0 && neighbor_id != id) {
          segments_set[{std::max(id, neighbor_id), std::min(id, neighbor_id)}]
              .emplace(i + dx, j + dy);
          segments_set[{std::max(id, neighbor_id), std::min(id, neighbor_id)}]
              .emplace(i, j);
        }
      }
    }
  }
  std::vector<std::vector<Coord<int>>> segments;
  for (const auto& [ids, pixel_coords_set] : segments_set) {
    constexpr size_t min_segment_size = 500;
    if (pixel_coords_set.size() >= min_segment_size) {
      std::vector<Coord<int>> pixel_coords_vector(pixel_coords_set.begin(),
                                                  pixel_coords_set.end());
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

auto SortSegments(std::vector<std::vector<Coord<int>>>& segments) {
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

auto GetMses(std::vector<std::vector<Coord<int>>>& segments)
    -> std::vector<std::vector<float>> {
  std::vector<std::vector<float>> mses;
  constexpr int window_size = 100;
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

auto GetCandidatesQuadCorners(
    const std::vector<std::vector<Coord<int>>>& segments,
    const std::vector<std::vector<float>>& mse_map)
    -> std::vector<CandidatesQuad> {
  std::vector<CandidatesQuad> quads;
  CHECK_EQ(mse_map.size(), segments.size());
  for (size_t idx = 0; idx < mse_map.size(); idx++) {
    const auto& segment = segments[idx];
    const auto& mse = mse_map[idx];
    CHECK_EQ(segment.size(), mse.size());
    constexpr int window_size = 100;
    CandidatesQuad quad{};
    std::array<float, quad.corners.size()> max_mse{};
    for (size_t i = 0; i < mse.size(); i++) {
      const float middle_mse = mse[(i + (window_size / 2)) % mse.size()];
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

void PopulateCandidateQuadCornersApriltagBuffer(
    std::vector<CandidatesQuad>& quads,
    ImageView candidates_quad_corners_apriltag) {
  for (const auto& quad : quads) {
    int color = 255;
    for (const auto& corner : quad.corners) {
      if (corner.row == 0 && corner.col == 0) {
        continue;
      }
      for (int i = -5; i <= 5; i++) {
        for (int j = -5; j <= 5; j++) {
          candidates_quad_corners_apriltag(corner.row + i, corner.col + j) =
              color;
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
    Coord<int> mean =
        std::accumulate(quad.corners.begin(), quad.corners.end(), Coord<int>{},
                        [](Coord<int> sum, Coord<int> value) -> Coord<int> {
                          sum.row += value.row;
                          sum.col += value.col;
                          return sum;
                        });
    mean.row /= 4;
    mean.col /= 4;
    std::ranges::sort(
        quad.corners, [&mean](Coord<int> a, Coord<int> b) -> bool {
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
        for (int j = -5; j <= 5; j++) {
          quad_apriltag(corner.row + i, corner.col + j) = color;
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
  for (int i = 0; i < 10; i++) {
    white += apriltag(bit_location[0][i].row, bit_location[0][i].col);
    white += apriltag(bit_location[9][i].row, bit_location[9][i].col);
    white += apriltag(bit_location[i][0].row, bit_location[i][0].col);
    white += apriltag(bit_location[i][9].row, bit_location[i][9].col);
  }
  white /= 40;

  float black = 0;
  for (int i = 1; i < 9; i++) {
    black += apriltag(bit_location[1][i].row, bit_location[1][i].col);
    black += apriltag(bit_location[8][i].row, bit_location[8][i].col);
    black += apriltag(bit_location[i][1].row, bit_location[i][1].col);
    black += apriltag(bit_location[i][8].row, bit_location[i][8].col);
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

      for (uint32_t k = 0; k < family->ncodes; k++) {
        int hamming = std::popcount(code ^ family->codes[k]);
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

auto DetectAprilTag(ImageView apriltag, bool imwrite)
    -> std::vector<ApriltagDetection> {
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
  if (imwrite) {
    ImWrite("/root/max.png", max);
    ImWrite("/root/min.png", min);
  }

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
  PopulateSegmentedApriltag(binarized_apriltag, segmented_apriltag);
  if (imwrite) {
    ImWrite("/root/segmented_apriltag.png", segmented_apriltag);
  }

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
  if (imwrite) {
    ImWrite("/root/candidate_quad_corners_apriltag.png",
            candidate_quad_corners_apriltag);
  }

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
  if (imwrite) {
    ImWrite("/root/quad_apriltag.png", quad_apriltag);
  }

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
  if (imwrite) {
    ImWrite("/root/bit_locations_apriltag.png", bit_locations_apriltag);
  }

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

  if (imwrite) {
    auto* refined_points_apriltag_buffer = static_cast<uint8_t*>(
        calloc(apriltag.width * apriltag.height, sizeof(uint8_t)));
    ImageView refined_points_apriltag{.data = refined_points_apriltag_buffer,
                                      .stride = apriltag.stride,
                                      .height = apriltag.height,
                                      .width = apriltag.width};
    memcpy(refined_points_apriltag_buffer,
           sorted_boundary_segmented_apriltag_buffer,
           sizeof(uint8_t) * apriltag.width * apriltag.height);

    auto refined_points = GetRefinedPoints(detections, apriltag);
    PopulateRefinedPointsApriltag(refined_points, refined_points_apriltag);
    ImWrite("/root/refined_points_apriltag.png", refined_points_apriltag);
    free(refined_points_apriltag_buffer);
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

}  // namespace apriltag
