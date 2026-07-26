#pragma once

#include <array>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include <Eigen/Core>
#include <frc/apriltag/AprilTagFieldLayout.h>
#include <frc/geometry/Pose3d.h>
#include <opencv2/core/mat.hpp>

#include "control_loop/node.h"
#include "localization/position.h"
#include "localization/unambiguous_solver_node.h"
#include "utils/solver_common.h"

namespace localization {

class JointSolverNode final : public control_loop::INode {
 public:
  JointSolverNode(std::string_view output_channel,
                  const std::vector<camera_constant_t>& camera_constants,
                  const frc::AprilTagFieldLayout& layout = kApriltagLayout);

  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback)
      override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

  // Refines starting_pose using all detections from all cameras jointly.
  auto Solve(const std::vector<std::vector<tag_detection_t>>& detection_batches,
             const frc::Pose3d& starting_pose,
             bool reject_far_tags = true) const
      -> std::optional<position_estimate_t>;

 private:
  struct CameraMatrices {
    Eigen::Matrix<double, 3, 4> image_to_robot;
  };

  struct DataPoint {
    Eigen::Vector2d image_point;
    const CameraMatrices* camera;
    Eigen::Vector4d field_to_tag_corner;
  };

  static void ComputeResidual(
      const std::vector<DataPoint>& data_points,
      const Eigen::Matrix4d& robot_to_field, Eigen::VectorXd& residual,
      Eigen::MatrixXd* jacobian = nullptr);

  std::string output_channel_;
  std::string seed_output_channel_;
  std::vector<std::string> detection_batch_channels_;
  std::vector<CameraMatrices> camera_matrices_;
  std::unordered_map<int, std::array<Eigen::Vector4d, 4>> tag_corners_;
  UnambiguousSolverNode seed_solver_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace localization
