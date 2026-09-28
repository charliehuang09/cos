#pragma once
#include <array>
#include "src/camera/camera_constants.h"

namespace gamepiece {
using lane_segment_t = struct LaneSegment {
  cv::Vec4f origin;
  cv::Vec4f end;
};

class LaneDensityTracker {
  static constexpr int num_lanes =
      2;  // actually 2x because this is reflected across center line
 public:
  using image_lane_segment_t = struct ImageLaneSegment {
    cv::Vec2f origin;
    cv::Vec2f end;
    cv::Vec2f midpoint;
    bool valid = false;
  };
  using line_t = struct Line {
    float a, b, c;
    [[nodiscard]] auto pointInNormalDirection(const cv::Vec2f& point) const
        -> bool {
      return (a * point(0) + b * point(1) + c) > 0;
    }
  };

  LaneDensityTracker(const camera::camera_constant_t& camera);
  auto GetImageLaneBoundaries(const frc::Pose3d& robot_pose)
      -> std::array<image_lane_segment_t, 2 * num_lanes + 1>;
  auto GetLaneDensities(const cv::Mat& color_image,
                        const frc::Pose3d& robot_pose)
      -> std::array<float, 2 * num_lanes>;

 private:
  cv::Matx44f camera_to_robot_cv_;
  cv::Matx33f camera_to_image_;
  cv::Vec<double, 5> distortion_coeffs_;
  // meters, wpilib coordinates
  static constexpr float lane_width = 1.0;
  static constexpr float center_field_x = 8.256524;
  static constexpr float field_width = 8.07;
  static constexpr float lane_begin_y = 1.0;
  inline static const cv::Matx34f Pi = cv::Matx34f(  // clang-format off
         1.0, 0.0, 0.0, 0.0,
         0.0, 1.0, 0.0, 0.0,
         0.0, 0.0, 1.0, 0.0);  // clang-format on
  // meters, opencv coordinates
  inline static const cv::Vec4f field_relative_lane_direction{
      -(field_width - lane_begin_y * 2), 0, 0, 0};
  inline static const cv::Vec4f field_relative_interlane_offset{0, 0,
                                                                lane_width, 0};
  inline static const cv::Vec4f field_relative_center_lane_origin{
      -lane_begin_y, 0, center_field_x, 1};
  inline static const std::array<lane_segment_t, 2 * num_lanes + 1>
      field_relative_lane_boundaries_ = [] {
        std::array<lane_segment_t, 2 * num_lanes + 1> lanes{};

        for (int offset = -num_lanes; offset <= num_lanes; ++offset) {
          const cv::Vec4f origin =
              field_relative_center_lane_origin +
              static_cast<float>(offset) * field_relative_interlane_offset;
          lanes[offset + num_lanes] = {
              .origin = origin,
              .end = origin + field_relative_lane_direction,
          };
        }

        return lanes;
      }();
};
}  // namespace gamepiece
