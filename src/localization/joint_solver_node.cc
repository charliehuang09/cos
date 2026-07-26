#include "localization/joint_solver_node.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <utility>

#include <unsupported/Eigen/MatrixFunctions>

#include "absl/log/log.h"
#include "utils/camera_config.h"
#include "utils/cv_geometry.h"
#include "utils/json.h"

namespace localization {
namespace {

constexpr int kMaxEpochs = 10'000;
constexpr int kMaxDampingSteps = 40;
constexpr double kAcceptableMeanSquaredError = 1e-8;
constexpr double kMaxAcceptableError = 1e5;
constexpr double kMaximumLambda = 1e10;
constexpr double kLambdaScalar = 2.0;

}  // namespace

JointSolverNode::JointSolverNode(
    std::string_view output_channel,
    const std::vector<camera_constant_t>& camera_constants,
    const frc::AprilTagFieldLayout& layout)
    : output_channel_(output_channel),
      seed_output_channel_(std::string(output_channel) + "/joint_seed"),
      seed_solver_(seed_output_channel_, camera_constants, layout),
      publications_({control_loop::MessageDescriptor(
          output_channel_, typeid(PositionEstimate))}) {
  detection_batch_channels_.reserve(camera_constants.size());
  camera_matrices_.reserve(camera_constants.size());

  Eigen::Matrix<double, 3, 4> projection =
      Eigen::Matrix<double, 3, 4>::Zero();
  projection.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity();

  for (const auto& constants : camera_constants) {
    const std::string channel = DetectionBatchChannel(constants.name);
    detection_batch_channels_.push_back(channel);

    const auto intrinsics = utils::ReadJson(constants.intrinsics_path);
    const cv::Mat camera_matrix_cv = utils::CameraMatrixFromJson(intrinsics);
    const Eigen::Matrix3d camera_matrix =
        utils::CvMatToEigen<Eigen::Matrix3d>(camera_matrix_cv);
    const frc::Transform3d camera_to_robot_transform =
        utils::ExtrinsicsJsonToCameraToRobot(
            utils::ReadJson(constants.extrinsics_path));
    Eigen::Matrix4d camera_to_robot = camera_to_robot_transform.ToMatrix();
    utils::ChangeBasis(camera_to_robot, utils::Basis::kWpiToCv);
    camera_matrices_.push_back(
        {.image_to_robot =
             camera_matrix * projection * camera_to_robot});
  }

  // The detector's corner ordering views the tag after a 180 degree yaw.
  Eigen::Matrix4d rotate_tag_yaw_cv = Eigen::Matrix4d::Identity();
  rotate_tag_yaw_cv(0, 0) = -1.0;
  rotate_tag_yaw_cv(2, 2) = -1.0;
  for (const auto& tag : layout.GetTags()) {
    Eigen::Matrix4d field_to_tag = tag.pose.ToMatrix();
    utils::ChangeBasis(field_to_tag, utils::Basis::kWpiToCv);
    std::array<Eigen::Vector4d, 4> corners;
    for (size_t corner = 0; corner < kApriltagCorners.size(); ++corner) {
      const cv::Point3d& point = kApriltagCorners[corner];
      corners[corner] =
          field_to_tag * rotate_tag_yaw_cv *
          utils::Homogenize(Eigen::Vector3d(point.x, point.y, point.z));
    }
    tag_corners_.emplace(tag.ID, std::move(corners));
  }

  seed_solver_.RegisterCallback(
      [this](const control_loop::Context& context) {
        const auto* seed =
            context->GetMessage<PositionEstimate>(seed_output_channel_);
        if (seed == nullptr) {
          return;
        }

        std::vector<std::vector<tag_detection_t>> batches;
        batches.reserve(detection_batch_channels_.size());
        for (const std::string& channel : detection_batch_channels_) {
          const auto* detections =
              context->GetMessage<apriltag::TagDetections>(channel);
          batches.push_back(detections == nullptr
                                ? std::vector<tag_detection_t>{}
                                : detections->tag_detections);
        }

        auto estimate = Solve(batches, seed->pose);
        if (!estimate.has_value()) {
          estimate = *seed;
        } else {
          // Keep the seed solver's distance-based uncertainty and rejected-tag
          // metadata; the joint solver's loss describes the reprojection fit.
          estimate->variance = seed->variance;
          estimate->rejected_tag_ids = seed->rejected_tag_ids;
        }

        context->SetMessage(
            output_channel_,
            std::make_unique<PositionEstimate>(std::move(*estimate)));

        for (const auto& callback : callbacks_) {
          callback(context);
        }
      });
}

void JointSolverNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto JointSolverNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return seed_solver_.CreateCallback();
}

auto JointSolverNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return seed_solver_.GetDependencies();
}

auto JointSolverNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void JointSolverNode::ComputeResidual(
    const std::vector<DataPoint>& data_points,
    const Eigen::Matrix4d& robot_to_field, Eigen::VectorXd& residual,
    Eigen::MatrixXd* d_residual_d_twist_jacobian) {
  if (d_residual_d_twist_jacobian == nullptr) {
    return;
  }
  for (size_t index = 0; index < data_points.size(); ++index) {
    const DataPoint& data_point = data_points[index];
    const Eigen::Vector4d robot_relative_corner =
        robot_to_field * data_point.field_to_tag_corner;
    Eigen::Vector3d projection =
        data_point.camera->image_to_robot * robot_relative_corner;
    const double depth = projection.z(); // notated as lambda

    if (!std::isfinite(depth) || std::abs(depth) < 1e-12) {
      residual.segment<2>(2 * index).setConstant(
          std::numeric_limits<double>::max() / 4.0);
      if (d_residual_d_twist_jacobian != nullptr) {
        d_residual_d_twist_jacobian->block<2, 6>(2 * index, 0).setZero();
      }
      continue;
    }

    projection /= depth;
    residual(2 * index) =
        data_point.image_point.x() - projection.x();
    residual(2 * index + 1) =
        data_point.image_point.y() - projection.y();

    Eigen::Matrix<double, 2, 3> d_residual_i_d_projection_i;
    d_residual_i_d_projection_i << -1.0 / depth, 0.0,
        projection.x() / depth, 0.0, -1.0 / depth,
        projection.y() / depth;

    Eigen::Matrix<double, 3, 6> d_robot_relative_object_point_d_d_twist;
    d_robot_relative_object_point_d_d_twist.block<3, 3>(0, 0) =
        -utils::CrossProduct(robot_relative_corner.head<3>());
    d_robot_relative_object_point_d_d_twist.block<3, 3>(0, 3) = Eigen::Matrix3d::Identity();

    d_residual_d_twist_jacobian->block<2, 6>(2 * index, 0) =
        d_residual_i_d_projection_i *
        data_point.camera->image_to_robot.block<3, 3>(0, 0) *
        d_robot_relative_object_point_d_d_twist;
  }
}

auto JointSolverNode::Solve(
    const std::vector<std::vector<tag_detection_t>>& detection_batches,
    const frc::Pose3d& starting_pose, bool reject_far_tags) const
    -> std::optional<position_estimate_t> {
  std::vector<DataPoint> data_points;
  std::vector<int> tag_ids;
  std::vector<int> rejected_tag_ids;

  const size_t camera_count =
      std::min(detection_batches.size(), camera_matrices_.size());
  for (size_t camera = 0; camera < camera_count; ++camera) {
    for (const tag_detection_t& detection : detection_batches[camera]) {
      const auto tag = tag_corners_.find(detection.tag_id);
      if (tag == tag_corners_.end()) {
        LOG(WARNING) << "Invalid tag id: " << detection.tag_id;
        continue;
      }
      if (reject_far_tags &&
          utils::QuadAreaPixels(detection.corners) < kMinTagAreaPixels) {
        rejected_tag_ids.push_back(detection.tag_id);
        continue;
      }

      tag_ids.push_back(detection.tag_id);
      for (size_t corner = 0; corner < detection.corners.size(); ++corner) {
        data_points.push_back(
            {.image_point =
                 {detection.corners[corner].x,
                  detection.corners[corner].y},
             .camera = &camera_matrices_[camera],
             .field_to_tag_corner = tag->second[corner]});
      }
    }
  }

  if (data_points.empty()) {
    return std::nullopt;
  }

  Eigen::Matrix4d robot_to_field = starting_pose.ToMatrix().inverse();
  utils::ChangeBasis(robot_to_field, utils::Basis::kWpiToCv);
  Eigen::VectorXd residual(2 * data_points.size());
  Eigen::MatrixXd jacobian(2 * data_points.size(), 6);
  double current_error = std::numeric_limits<double>::infinity();

  for (int epoch = 0; epoch < kMaxEpochs; ++epoch) {
    ComputeResidual(data_points, robot_to_field, residual, &jacobian);
    current_error = residual.squaredNorm();
    if (!std::isfinite(current_error)) {
      return std::nullopt;
    }
    if (current_error / static_cast<double>(data_points.size()) <=
        kAcceptableMeanSquaredError) {
      break;
    }

    const Eigen::Matrix<double, 6, 6> hessian =
        jacobian.transpose() * jacobian;
    const Eigen::Vector<double, 6> gradient =
        -jacobian.transpose() * residual;
    Eigen::Matrix<double, 6, 6> hessian_diagonal =
        hessian.diagonal().asDiagonal();
    for (int diagonal = 0; diagonal < 6; ++diagonal) {
      hessian_diagonal(diagonal, diagonal) =
          std::max(hessian_diagonal(diagonal, diagonal), 1e-12);
    }

    bool accepted = false;
    double lambda = 1.0;
    for (int attempt = 0; attempt < kMaxDampingSteps &&
                          lambda <= kMaximumLambda;
         ++attempt, lambda *= kLambdaScalar) {
      const Eigen::Vector<double, 6> partial_twist =
          (hessian + lambda * hessian_diagonal).ldlt().solve(gradient);
      if (!partial_twist.allFinite()) {
        continue;
      }

      Eigen::Matrix4d expanded_twist = Eigen::Matrix4d::Zero();
      expanded_twist.block<3, 3>(0, 0) =
          utils::CrossProduct(partial_twist.head<3>());
      expanded_twist.block<3, 1>(0, 3) = partial_twist.tail<3>();
      const Eigen::Matrix4d candidate =
          expanded_twist.exp() * robot_to_field;

      ComputeResidual(data_points, candidate, residual);
      const double candidate_error = residual.squaredNorm();
      if (std::isfinite(candidate_error) &&
          candidate_error < current_error) {
        robot_to_field = candidate;
        current_error = candidate_error;
        accepted = true;
        break;
      }
    }

    if (!accepted) {
      break;
    }
  }

  ComputeResidual(data_points, robot_to_field, residual);
  current_error = residual.squaredNorm();
  if (!std::isfinite(current_error) ||
      current_error > kMaxAcceptableError) {
    return std::nullopt;
  }

  Eigen::Matrix4d field_to_robot = robot_to_field.inverse();
  field_to_robot.row(3) << 0.0, 0.0, 0.0, 1.0;
  utils::ChangeBasis(field_to_robot, utils::Basis::kCvToWpi);

  position_estimate_t estimate;
  estimate.tag_ids = std::move(tag_ids);
  estimate.rejected_tag_ids = std::move(rejected_tag_ids);
  estimate.pose = frc::Pose3d(field_to_robot);
  estimate.num_tags = static_cast<int>(estimate.tag_ids.size());
  estimate.loss = current_error;
  return estimate;
}

}  // namespace localization
