#pragma once

#include <cstddef>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "control_loop/node.h"
#include "localization/multi_tag_solver_node.h"
#include "localization/position.h"
#include "localization/solver_common.h"

namespace localization {

class UnambiguousSolverNode final : public control_loop::INode {
 public:
  UnambiguousSolverNode(std::vector<std::string> input_channels,
                        std::string_view output_channel,
                        const std::vector<camera_constant_t>& camera_constants,
                        const wpi::apriltag::AprilTagFieldLayout& layout =
                            kApriltagLayout);

  void RegisterCallback(
      std::function<void(const control_loop::Context&)> callback) override;
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;

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
  void HandleCameraInput(const control_loop::Context& context,
                         size_t camera_id);

  struct PendingSolve {
    size_t num_cameras_received = 0;
    bool completed = false;
  };

  std::string output_channel_;
  std::vector<std::string> input_channels_;
  size_t num_cameras_ = 0;
  std::vector<MultiTagSolverNode> multitag_solvers_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::mutex pending_solves_mutex_;
  std::unordered_map<const control_loop::ContextInternal*, PendingSolve>
      pending_solves_;
  std::optional<position_estimate_t> prev_pose_estimate_;
};

}  // namespace localization
