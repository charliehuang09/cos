#include <opencv2/opencv.hpp>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/log/log.h"

using std::pair;

const std::array<int, 4> dx_array = {0, 0, 1, -1};
const std::array<int, 4> dy_array = {-1, 1, 0, 0};

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
        binarized_apriltag(i, j) = 255 / 2;
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
      if (binarized_apriltag(i, j) != 255 / 2 &&
          segmented_apriltag(i, j) == 0) {
        Segment(i, j, binarized_apriltag, segmented_apriltag, id++);
      }
    }
  }
}
auto GetSegments(ImageView32 segmented_apriltag)
    -> absl::flat_hash_map<std::pair<uint, uint>,
                           std::vector<std::pair<uint, uint>>> {
  absl::flat_hash_map<std::pair<uint, uint>, std::vector<std::pair<uint, uint>>>
      segments;
  for (uint i = 1; i < segmented_apriltag.height - 1; i++) {
    for (uint j = 1; j < segmented_apriltag.width - 1; j++) {
      if (segmented_apriltag(i, j) != 0) {
        auto id = segmented_apriltag(i, j);
        for (const auto& dx : dx_array) {
          for (const auto& dy : dy_array) {
            auto neighbor_id = segmented_apriltag(i + dx, j + dy);
            if (neighbor_id != 0 && neighbor_id != id) {
              segments[{std::max(id, neighbor_id), std::min(id, neighbor_id)}]
                  .emplace_back(i + dx, j + dy);
            }
          }
        }
      }
    }
  }
  return segments;
}

void PopulateBoundarySegmentedApriltag(
    absl::flat_hash_map<std::pair<uint, uint>,
                        std::vector<std::pair<uint, uint>>>& segments,
    ImageView32 boundary_segmented_apriltag) {
  for (const auto& [ids, pixel_coords] : segments) {
    for (const auto& pixel_coord : pixel_coords) {
      boundary_segmented_apriltag(pixel_coord.first, pixel_coord.second) =
          ids.first + ids.second;
    }
  }
}

void DetectAprilTag(ImageView apriltag) {
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

  auto* boundary_segmented_apriltag_buffer = static_cast<uint32_t*>(
      calloc(apriltag.width * apriltag.height, sizeof(uint32_t)));
  ImageView32 boundary_segmented_apriltag{
      .data = boundary_segmented_apriltag_buffer,
      .stride = apriltag.stride,
      .height = apriltag.height,
      .width = apriltag.width};
  PopulateBoundarySegmentedApriltag(segments, boundary_segmented_apriltag);
  ImWrite("/root/boundary_segmented_apriltag.png", boundary_segmented_apriltag);

  free(min_buffer);
  free(max_buffer);
  free(threshold_buffer);
  free(valid_buffer);
  free(binarized_apriltag_buffer);
  free(boundary_segmented_apriltag_buffer);
}

auto main() -> int {
  std::string apriltag_path = "/root/apriltag.png";
  cv::Mat apriltag = cv::imread(apriltag_path, cv::IMREAD_GRAYSCALE);
  uint8_t* pixels = apriltag.data;
  uint height = apriltag.rows;
  uint width = apriltag.cols;
  CHECK(apriltag.step == static_cast<size_t>(apriltag.cols));
  DetectAprilTag(ImageView{
      .data = pixels, .stride = width, .height = height, .width = width});
}
