#include <opencv2/opencv.hpp>

#include "absl/log/check.h"
#include "absl/log/log.h"

struct ImageView {
  uint8_t* data;
  uint stride;
  uint height;
  uint width;

  auto operator()(size_t row, size_t col) -> uint8_t& {
    return data[row * stride + col];
  }
};

void ImWrite(const std::string& path, const ImageView& image) {
  cv::Mat mat(image.height, image.width, CV_8UC1, image.data, image.stride);
  cv::imwrite(path, mat);
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
                           ImageView bineriazed_apriltag, uint row, uint col) {
  row *= 4;
  col *= 4;
  if (valid == 0) {
    for (uint i = row; i < row + 4; i++) {
      for (uint j = col; j < col + 4; j++) {
        bineriazed_apriltag(i, j) = 255 / 2;
      }
    }
  } else {
    for (uint i = row; i < row + 4; i++) {
      for (uint j = col; j < col + 4; j++) {
        bineriazed_apriltag(i, j) = (apriltag(i, j) > threshold) ? 255 : 0;
      }
    }
  }
}

void PopulateBinarizedApriltag(ImageView threshold, ImageView valid,
                               ImageView apriltag,
                               ImageView bineriazed_apriltag) {
  CHECK_EQ(threshold.height * 4, apriltag.height);
  CHECK_EQ(threshold.width * 4, apriltag.width);
  CHECK_EQ(threshold.height * 4, bineriazed_apriltag.height);
  CHECK_EQ(threshold.width * 4, bineriazed_apriltag.width);
  CHECK_EQ(threshold.height, valid.height);
  CHECK_EQ(threshold.width, valid.width);

  for (uint j = 0; j < threshold.width; j++) {
    for (uint i = 0; i < threshold.height; i++) {
      ApplyThreshold(threshold(i, j), valid(i, j), apriltag,
                     bineriazed_apriltag, i, j);
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

  free(min_buffer);
  free(max_buffer);
  free(threshold_buffer);
  free(valid_buffer);
  free(binarized_apriltag_buffer);
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
