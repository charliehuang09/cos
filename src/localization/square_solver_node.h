#pragma once

#include <functional>
#include <string>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <wpi/apriltag/AprilTagFieldLayout.hpp>
#include <wpi/math/geometry/Pose3d.hpp>

#include "control_loop/node.h"
#include "utils/solver_common.h"

namespace localization {

class SquareSolverNode final : public control_loop::INode {
 public:
  SquareSolverNode(std::string_view input_channel,
                   std::string_view output_channel,
                   const std::string& intrinsics_path,
                   const std::string& extrinsics_path,
                   wpi::apriltag::AprilTagFieldLayout layout =
                       kApriltagLayout,
                   std::vector<cv::Point3d> tag_corners = kApriltagCorners);

  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback)
      override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

  auto AmbiguousSolve(
      const std::vector<tag_detection_t>& detections,
      bool reject_far_tags = true) -> std::vector<ambiguous_estimate_t>;

 private:
  static constexpr double kVarianceScalar = 1.0;
  static constexpr double kVarianceMin = 0.0;

  std::string input_channel_;
  std::string output_channel_;
  wpi::apriltag::AprilTagFieldLayout layout_;
  std::vector<cv::Point3d> tag_corners_;
  cv::Mat camera_matrix_;
  cv::Mat distortion_coefficients_;
  cv::Mat camera_to_robot_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace localization
