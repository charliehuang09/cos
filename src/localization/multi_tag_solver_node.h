#pragma once

#include <functional>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <wpi/apriltag/AprilTagFieldLayout.hpp>

#include "control_loop/node.h"
#include "utils/solver_common.h"
#include "localization/square_solver_node.h"

namespace localization {

class MultiTagSolverNode final : public control_loop::INode {
 public:
  MultiTagSolverNode(std::string_view input_channel,
                     std::string_view output_channel,
                     const std::string& intrinsics_path,
                     const std::string& extrinsics_path,
                     const wpi::apriltag::AprilTagFieldLayout& layout =
                         kApriltagLayout,
                     const std::vector<cv::Point3d>& tag_corners =
                         kApriltagCorners);

  void RegisterCallback(
      std::function<void(const control_loop::Context&)> callback) override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;

  auto AmbiguousSolve(
      const std::vector<tag_detection_t>& detections,
      bool reject_far_tags = true) -> std::optional<ambiguous_estimate_t>;

 private:
  static constexpr double kVarianceScalar = 0.7;
  static constexpr double kVarianceMin = 1.0;

  std::string input_channel_;
  std::string output_channel_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_coefficients_;
  cv::Mat camera_to_robot_;
  std::unordered_map<int, std::array<cv::Point3d, 4>> tag_corners_;
  SquareSolverNode single_tag_solver_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace localization
