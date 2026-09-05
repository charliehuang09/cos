#include "localization/square_solver_node.h"

#include <utility>

#include <opencv2/calib3d.hpp>

#include "absl/log/log.h"
#include "utils/cv_geometry.h"

namespace localization {

SquareSolverNode::SquareSolverNode(std::string_view input_channel,
                                   std::string_view output_channel,
                                   const camera::Intrinsics& intrinsics,
                                   const camera::Extrinsics& extrinsics,
                                   frc::AprilTagFieldLayout layout,
                                   std::vector<cv::Point3d> tag_corners)
    : input_channel_(input_channel),
      output_channel_(output_channel),
      layout_(std::move(layout)),
      tag_corners_(std::move(tag_corners)),
      camera_matrix_(intrinsics.ToMatrix()),
      distortion_coefficients_(intrinsics.ToDistortionCoefficients()),
      camera_to_robot_(extrinsics.ToCameraToRobot<cv::Mat>()),
      dependencies_({control_loop::MessageDescriptor(
          input_channel_, typeid(apriltag::TagDetections))}),
      publications_({control_loop::MessageDescriptor(
          output_channel_, typeid(AmbiguousEstimateMessage))}) {}

void SquareSolverNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.push_back(callback);
}

auto SquareSolverNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto notify_callbacks = [this, &context]() -> void {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };

    auto* detections =
        context->GetMessage<apriltag::TagDetections>(input_channel_);
    if (detections == nullptr) {
      notify_callbacks();
      return;
    }

    auto estimates = AmbiguousSolve(detections->tag_detections);
    if (estimates.empty()) {
      LOG(WARNING) << "Square solver produced no pose estimates";
      context->SetMessage(output_channel_, nullptr);
    } else {
      context->SetMessage(
          output_channel_,
          std::make_unique<AmbiguousEstimateMessage>(std::move(estimates)));
    }
    notify_callbacks();
  };
}

auto SquareSolverNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto SquareSolverNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
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
                              const cv::Mat& tvec) -> solver_estimate_t {
      const double distance = cv::norm(tvec);
      solver_estimate_t estimate;
      estimate.tag_ids = {detection.tag_id};
      estimate.distances = {distance};
      estimate.pose = utils::ComputeRobotPose(tvec, rvec, detection.tag_id,
                                              layout_, camera_to_robot_);
      estimate.variance = Variance(1, distance, kVarianceMin, kVarianceScalar);
      estimate.distance = distance;
      return estimate;
    };

    auto est1 = build_estimate(rvecs[0], tvecs[0]);
    auto est2 = build_estimate(rvecs[1], tvecs[1]);
    if (reject_far_tags && est1.distance > kMaxTagDistance &&
        est2.distance > kMaxTagDistance) {
      continue;
    }

    pose_estimates.push_back(
        {.pos1 = std::move(est1), .pos2 = std::move(est2)});
  }
  return pose_estimates;
}

}  // namespace localization
