#include "localization/square_solver_node.h"

#include <utility>

#include <opencv2/calib3d.hpp>

#include "utils/camera_config.h"
#include "utils/cv_geometry.h"
#include "utils/json.h"

namespace localization {

SquareSolverNode::SquareSolverNode(
    std::string_view input_channel, std::string_view output_channel,
    const std::string& intrinsics_path, const std::string& extrinsics_path,
    wpi::apriltag::AprilTagFieldLayout layout,
    std::vector<cv::Point3d> tag_corners)
    : input_channel_(input_channel),
      output_channel_(output_channel),
      layout_(std::move(layout)),
      tag_corners_(std::move(tag_corners)),
      camera_matrix_(
          utils::CameraMatrixFromJson(utils::ReadJson(intrinsics_path))),
      distortion_coefficients_(utils::DistortionCoefficientsFromJson(
          utils::ReadJson(intrinsics_path))),
      camera_to_robot_(utils::EigenToCvMat(
          utils::ExtrinsicsJsonToCameraToRobot(
              utils::ReadJson(extrinsics_path))
              .ToMatrix())) {}

void SquareSolverNode::RegisterCallback(
    std::function<void(const control_loop::Context&)> callback) {
  callbacks_.push_back(std::move(callback));
}

auto SquareSolverNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    auto* detections =
        context->GetMessage<apriltag::NvidiaTagDetections>(input_channel_);
    if (detections == nullptr) {
      auto* failed = context->GetMessage<control_loop::FailedMessage>(
          input_channel_);
      if (failed != nullptr) {
        context->SetMessage(
            output_channel_,
            std::make_unique<control_loop::FailedMessage>(
                output_channel_, "Upstream failed: " + failed->reason));
        for (const auto& callback : callbacks_) {
          callback(context);
        }
      }
      return;
    }

    auto estimates = AmbiguousSolve(detections->tag_detections);
    if (estimates.empty()) {
      context->SetMessage(
          output_channel_,
          std::make_unique<control_loop::FailedMessage>(
              output_channel_, "Square solver produced no pose estimates"));
    } else {
      context->SetMessage(output_channel_,
                          std::make_unique<AmbiguousEstimateMessage>(
                              std::move(estimates)));
    }
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

auto SquareSolverNode::AmbiguousSolve(
    const std::vector<tag_detection_t>& detections, bool reject_far_tags)
    -> std::vector<ambiguous_estimate_t> {
  std::vector<ambiguous_estimate_t> pose_estimates;
  for (const auto& detection : detections) {
    if (reject_far_tags &&
        utils::QuadAreaPixels(detection.corners) < kMinTagAreaPixels) {
      continue;
    }

    std::vector<cv::Mat> rvecs;
    std::vector<cv::Mat> tvecs;
    cv::solvePnPGeneric(tag_corners_, detection.corners, camera_matrix_,
                        distortion_coefficients_, rvecs, tvecs, false,
                        cv::SOLVEPNP_IPPE_SQUARE);

    if (rvecs.size() < 2 || tvecs.size() < 2) {
      continue;
    }

    auto build_estimate = [&](const cv::Mat& rvec,
                              const cv::Mat& tvec) -> position_estimate_t {
      const double distance = cv::norm(tvec);
      position_estimate_t estimate;
      estimate.tag_ids = {detection.tag_id};
      estimate.pose = utils::ComputeRobotPose(tvec, rvec, detection.tag_id,
                                              layout_, camera_to_robot_);
      estimate.variance = Variance(1, distance, kVarianceMin, kVarianceScalar);
      estimate.num_tags = 1;
      estimate.avg_tag_dist = distance;
      return estimate;
    };

    auto est1 = build_estimate(rvecs[0], tvecs[0]);
    auto est2 = build_estimate(rvecs[1], tvecs[1]);
    if (reject_far_tags && est1.avg_tag_dist > kMaxTagDistance &&
        est2.avg_tag_dist > kMaxTagDistance) {
      continue;
    }

    pose_estimates.push_back({.pos1 = std::move(est1),
                              .pos2 = std::move(est2)});
  }
  return pose_estimates;
}

}  // namespace localization
