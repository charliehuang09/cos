#include "localization/unambiguous_solver_node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include <wpi/math/geometry/Quaternion.hpp>
#include <wpi/math/geometry/Rotation3d.hpp>

namespace localization {

UnambiguousSolverNode::UnambiguousSolverNode(
    std::vector<std::string> input_channels, std::string_view output_channel,
    const std::vector<camera_constant_t>& camera_constants,
    const wpi::apriltag::AprilTagFieldLayout& layout)
    : output_channel_(output_channel),
      input_channels_(std::move(input_channels)),
      num_cameras_(camera_constants.size()) {
  multitag_solvers_.reserve(camera_constants.size());
  for (size_t camera_id = 0; camera_id < camera_constants.size();
       ++camera_id) {
    const camera_constant_t& constants = camera_constants[camera_id];
    multitag_solvers_.emplace_back(input_channels_.at(camera_id),
                                   output_channel_, constants.intrinsics_path,
                                   constants.extrinsics_path, layout);
  }
}

void UnambiguousSolverNode::RegisterCallback(
    std::function<void(const control_loop::Context&)> callback) {
  callbacks_.push_back(std::move(callback));
}

auto UnambiguousSolverNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  throw std::logic_error(
      "UnambiguousSolverNode requires CreateCallback(camera_id)");
}

void UnambiguousSolverNode::HandleCameraInput(
    const control_loop::Context& context, size_t camera_id) {
  if (camera_id >= num_cameras_) {
    throw std::out_of_range("UnambiguousSolverNode camera_id out of range");
  }

  if (context->GetMessage<control_loop::IMessage>(output_channel_) !=
      nullptr) {
    return;
  }

  std::vector<std::vector<tag_detection_t>> detection_batches;
  {
    std::lock_guard lock(pending_solves_mutex_);
    PendingSolve& pending = pending_solves_[context.get()];
    if (pending.completed) {
      return;
    }
    ++pending.num_cameras_received;

    if (pending.num_cameras_received < num_cameras_) {
      return;
    }
    pending.completed = true;
    detection_batches.reserve(num_cameras_);
    for (const std::string& input_channel : input_channels_) {
      auto* maybe_detection_batch =
          context->GetMessage<control_loop::IMessage>(input_channel);
      if (maybe_detection_batch->GetType() == typeid(control_loop::FailedMessage)) {
        detection_batches.emplace_back();
        continue;
      }

      auto detection_batch = static_cast<apriltag::NvidiaTagDetections*>(maybe_detection_batch);
      detection_batches.push_back(detection_batch->tag_detections);
    }
  }

  auto result = Solve(detection_batches);
  if (!result.has_value()) {
    context->SetMessage(
        output_channel_,
        std::make_unique<control_loop::FailedMessage>(
            output_channel_,
            "Unambiguous solver produced no position estimate"));
  } else {
    context->SetMessage(
        output_channel_,
        std::make_unique<PositionEstimate>(std::move(*result)));
  }
  for (const auto& callback : callbacks_) {
    callback(context);
  }

  std::lock_guard lock(pending_solves_mutex_);
  pending_solves_.erase(context.get());
}

auto UnambiguousSolverNode::Cost(const wpi::math::Pose3d& a,
                                 const wpi::math::Pose3d& b) -> double {
  const double translation = a.Translation().Distance(b.Translation()).value();
  const wpi::math::Rotation3d delta = a.Rotation().RelativeTo(b.Rotation());
  constexpr double kRotationWeight = 0.1;
  return translation + kRotationWeight * delta.Angle().value();
}

auto UnambiguousSolverNode::ComputeCost(
    const std::vector<position_estimate_t>& poses) -> double {
  double cost = 0.0;
  for (size_t i = 0; i < poses.size(); ++i) {
    if (poses[i].invalid) {
      return 1000.0;
    }
    for (size_t j = i + 1; j < poses.size(); ++j) {
      cost += Cost(poses[i].pose, poses[j].pose);
    }
    if (prev_pose_estimate_.has_value()) {
      cost += Cost(poses[i].pose, prev_pose_estimate_->pose);
    }
  }
  return cost;
}

auto UnambiguousSolverNode::WeightedAveragePose(
    const std::vector<position_estimate_t>& solutions) -> wpi::math::Pose3d {
  if (solutions.empty()) {
    return wpi::math::Pose3d{};
  }
  if (solutions.size() == 1) {
    return solutions.front().pose;
  }

  double total_weight = 0.0;
  for (const auto& estimate : solutions) {
    total_weight += 1.0 / estimate.variance;
  }

  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
  double qw = 0.0;
  double qx = 0.0;
  double qy = 0.0;
  double qz = 0.0;

  for (const auto& estimate : solutions) {
    const double weight = (1.0 / estimate.variance) / total_weight;
    x += weight * estimate.pose.X().value();
    y += weight * estimate.pose.Y().value();
    z += weight * estimate.pose.Z().value();

    auto quaternion = estimate.pose.Rotation().GetQuaternion();
    const double dot = qw * quaternion.W() + qx * quaternion.X() +
                       qy * quaternion.Y() + qz * quaternion.Z();
    const double sign = dot < 0.0 ? -1.0 : 1.0;
    qw += weight * sign * quaternion.W();
    qx += weight * sign * quaternion.X();
    qy += weight * sign * quaternion.Y();
    qz += weight * sign * quaternion.Z();
  }

  const double norm = std::sqrt(qw * qw + qx * qx + qy * qy + qz * qz);
  qw /= norm;
  qx /= norm;
  qy /= norm;
  qz /= norm;

  return wpi::math::Pose3d{
      wpi::units::meter_t{x}, wpi::units::meter_t{y}, wpi::units::meter_t{z},
      wpi::math::Rotation3d{wpi::math::Quaternion{qw, qx, qy, qz}}};
}

auto UnambiguousSolverNode::SearchSolutions(
    const std::vector<ambiguous_estimate_t>& all_pose_estimates, size_t index,
    std::vector<position_estimate_t>& current_solution,
    std::vector<position_estimate_t>& best_solution, double& best_cost)
    -> double {
  if (index == all_pose_estimates.size()) {
    const double cost = ComputeCost(current_solution);
    if (cost < best_cost) {
      best_cost = cost;
      best_solution = current_solution;
    }
    return best_cost;
  }

  const ambiguous_estimate_t& maybe_ambiguous = all_pose_estimates[index];
  current_solution.push_back(maybe_ambiguous.pos1);
  SearchSolutions(all_pose_estimates, index + 1, current_solution,
                  best_solution, best_cost);
  current_solution.pop_back();

  if (maybe_ambiguous.pos2.has_value()) {
    current_solution.push_back(*maybe_ambiguous.pos2);
    SearchSolutions(all_pose_estimates, index + 1, current_solution,
                    best_solution, best_cost);
    current_solution.pop_back();
  }
  return best_cost;
}

auto UnambiguousSolverNode::GetAmbiguousEstimates(
    const std::vector<std::vector<tag_detection_t>>& detection_batches,
    bool reject_far_tags) -> std::vector<ambiguous_estimate_t> {
  std::vector<ambiguous_estimate_t> estimates;
  const size_t num_cameras =
      std::min(multitag_solvers_.size(), detection_batches.size());
  for (size_t i = 0; i < num_cameras; ++i) {
    if (detection_batches[i].empty()) {
      continue;
    }
    auto estimate =
        multitag_solvers_[i].AmbiguousSolve(detection_batches[i],
                                            reject_far_tags);
    if (!estimate.has_value()) {
      continue;
    }

    const bool first_off_field = PoseOffField(estimate->pos1.pose);
    if (estimate->pos2.has_value()) {
      const bool second_off_field = PoseOffField(estimate->pos2->pose);
      if (first_off_field && second_off_field) {
        continue;
      }
      estimate->pos1.invalid = first_off_field;
      estimate->pos2->invalid = second_off_field;
    } else if (first_off_field) {
      continue;
    }

    estimates.push_back(std::move(*estimate));
  }
  return estimates;
}

auto UnambiguousSolverNode::Solve(
    const std::vector<std::vector<tag_detection_t>>& detection_batches,
    bool reject_far_tags) -> std::optional<position_estimate_t> {
  const auto ambiguous_estimates =
      GetAmbiguousEstimates(detection_batches, reject_far_tags);
  std::vector<position_estimate_t> best_solution;
  std::vector<position_estimate_t> current_solution;
  double best_cost = std::numeric_limits<double>::infinity();
  const double cost = SearchSolutions(ambiguous_estimates, 0, current_solution,
                                      best_solution, best_cost);

  if (best_solution.empty()) {
    return std::nullopt;
  }

  double avg_variance = 0.0;
  std::vector<int> tag_ids;
  for (const position_estimate_t& estimate : best_solution) {
    avg_variance += estimate.variance;
    tag_ids.insert(tag_ids.end(), estimate.tag_ids.begin(),
                   estimate.tag_ids.end());
  }
  avg_variance /= static_cast<double>(best_solution.size());
  const int num_tags = static_cast<int>(tag_ids.size());

  position_estimate_t estimate;
  estimate.tag_ids = std::move(tag_ids);
  estimate.pose = WeightedAveragePose(best_solution);
  estimate.variance = avg_variance;
  estimate.num_tags = num_tags;
  estimate.loss = cost;
  prev_pose_estimate_ = estimate;
  return estimate;
}

}  // namespace localization
