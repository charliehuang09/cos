#pragma once

#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "control_loop/node.h"
#include "localization/multi_tag_solver_node.h"
#include "localization/position.h"
#include "utils/solver_common.h"

namespace localization {

class UnambiguousSolverNode final : public control_loop::INode {
 public:
  UnambiguousSolverNode(std::string_view output_channel,
                        const std::vector<camera_constant_t>& camera_constants,
                        const wpi::apriltag::AprilTagFieldLayout& layout =
                            kApriltagLayout);

  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback)
      override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

  auto Solve(
      const std::vector<std::vector<tag_detection_t>>& detection_batches,
      bool reject_far_tags = true) -> std::optional<position_estimate_t>;

 private:
  static auto Cost(const wpi::math::Pose3d& a,
                   const wpi::math::Pose3d& b) -> double;
  auto ComputeCost(const std::vector<position_estimate_t>& poses) -> double;
  static auto WeightedAveragePose(
      const std::vector<position_estimate_t>& solutions) -> wpi::math::Pose3d;
  auto SearchSolutions(
      const std::vector<ambiguous_estimate_t>& all_pose_estimates,
      size_t index, std::vector<position_estimate_t>& current_solution,
      std::vector<position_estimate_t>& best_solution,
      double& best_cost) -> double;
  auto GetAmbiguousEstimates(
      const std::vector<std::vector<tag_detection_t>>& detection_batches,
      bool reject_far_tags) -> std::vector<ambiguous_estimate_t>;

  std::string output_channel_;
  std::vector<std::string> detection_batch_channels_;
  size_t ready_detection_batches_;
  const size_t expected_cameras_;
  std::vector<MultiTagSolverNode> multitag_solvers_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::optional<position_estimate_t> prev_pose_estimate_;
};

}  // namespace localization
