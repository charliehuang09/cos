#include "localization/multi_tag_solver_node.h"

#include <numbers>
#include <utility>

#include <opencv2/calib3d.hpp>

#include "absl/log/log.h"
#include "utils/camera_config.h"
#include "utils/cv_geometry.h"
#include "utils/json.h"

namespace localization {

MultiTagSolverNode::MultiTagSolverNode(
    std::string_view input_channel, std::string_view output_channel,
    const std::string& intrinsics_path, const std::string& extrinsics_path,
    const wpi::apriltag::AprilTagFieldLayout& layout,
    const std::vector<cv::Point3d>& tag_corners)
    : input_channel_(input_channel),
      output_channel_(output_channel),
      camera_matrix_(
          utils::CameraMatrixFromJson(utils::ReadJson(intrinsics_path))),
      distortion_coefficients_(utils::DistortionCoefficientsFromJson(
          utils::ReadJson(intrinsics_path))),
      camera_to_robot_(utils::Transform3dToCvMat(
          utils::ExtrinsicsJsonToCameraToRobot(
              utils::ReadJson(extrinsics_path)))),
      single_tag_solver_(input_channel, output_channel, intrinsics_path,
                         extrinsics_path, layout, tag_corners) {
  cv::Mat rvec = (cv::Mat_<double>(3, 1) << 0, std::numbers::pi, 0);
  cv::Mat tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  cv::Mat rotate_z = utils::MakeTransform(rvec, tvec);

  for (const wpi::apriltag::AprilTag& tag : layout.GetTags()) {
    cv::Mat field_to_tag = utils::Pose3dToCvMat(tag.pose);
    tag_corners_[tag.ID] = {
        utils::CvMatToPoint3d(field_to_tag * rotate_z *
                              utils::HomogenizePoint3d(tag_corners[0])),
        utils::CvMatToPoint3d(field_to_tag * rotate_z *
                              utils::HomogenizePoint3d(tag_corners[1])),
        utils::CvMatToPoint3d(field_to_tag * rotate_z *
                              utils::HomogenizePoint3d(tag_corners[2])),
        utils::CvMatToPoint3d(field_to_tag * rotate_z *
                              utils::HomogenizePoint3d(tag_corners[3])),
    };
  }
}

void MultiTagSolverNode::RegisterCallback(
    std::function<void(const control_loop::Context&)> callback) {
  callbacks_.push_back(std::move(callback));
}

auto MultiTagSolverNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) {
    auto notify_callbacks = [this, &context]() -> void {
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };

    auto* detections =
        context->GetMessage<apriltag::NvidiaTagDetections>(input_channel_);
    if (detections == nullptr) {
      notify_callbacks();
      return;
    }

    auto estimate = AmbiguousSolve(detections->tag_detections);
    if (!estimate.has_value()) {
      VLOG(1) << "Multi-tag solver produced no pose estimate";
    } else {
      std::vector<AmbiguousEstimate> estimates;
      estimates.push_back(std::move(*estimate));
      context->SetMessage(output_channel_,
                          std::make_unique<AmbiguousEstimateMessage>(
                              std::move(estimates)));
    }
    notify_callbacks();
  };
}

auto MultiTagSolverNode::AmbiguousSolve(
    const std::vector<tag_detection_t>& detections, bool reject_far_tags)
    -> std::optional<ambiguous_estimate_t> {
  std::vector<cv::Point3d> object_points;
  std::vector<cv::Point2d> image_points;
  std::vector<int> tag_ids;
  std::vector<int> rejected_tag_ids;
  std::vector<tag_detection_t> accepted_detections;
  double avg_distance = 0.0;

  for (const tag_detection_t& detection : detections) {
    auto tag_corners_it = tag_corners_.find(detection.tag_id);
    if (tag_corners_it == tag_corners_.end()) {
      LOG(WARNING) << "Invalid tag id: " << detection.tag_id;
      continue;
    }
    if (reject_far_tags &&
        utils::QuadAreaPixels(detection.corners) < kMinTagAreaPixels) {
      rejected_tag_ids.push_back(detection.tag_id);
      continue;
    }

    cv::Mat rvec_tag = cv::Mat::zeros(3, 1, CV_64FC1);
    cv::Mat tvec_tag = cv::Mat::zeros(3, 1, CV_64FC1);
    std::vector<cv::Point2d> corners(detection.corners.begin(),
                                     detection.corners.end());
    cv::solvePnP(kApriltagCorners, corners, camera_matrix_,
                 distortion_coefficients_, rvec_tag, tvec_tag, false,
                 cv::SOLVEPNP_IPPE_SQUARE);

    if (reject_far_tags && cv::norm(tvec_tag) > kMaxTagDistance) {
      rejected_tag_ids.push_back(detection.tag_id);
      continue;
    }

    avg_distance += cv::norm(tvec_tag);
    tag_ids.push_back(detection.tag_id);
    accepted_detections.push_back(detection);
    image_points.insert(image_points.end(), detection.corners.begin(),
                        detection.corners.end());
    object_points.insert(object_points.end(), tag_corners_it->second.begin(),
                         tag_corners_it->second.end());
  }

  if (image_points.empty() || object_points.empty()) {
    return std::nullopt;
  }

  if (tag_ids.size() == 1) {
    const std::vector<ambiguous_estimate_t> square_estimates =
        single_tag_solver_.AmbiguousSolve(accepted_detections,
                                          reject_far_tags);
    if (square_estimates.empty()) {
      return std::nullopt;
    }
    return square_estimates.front();
  }

  avg_distance /= static_cast<double>(tag_ids.size());
  cv::Mat rvec = cv::Mat::zeros(3, 1, CV_64FC1);
  cv::Mat tvec = cv::Mat::zeros(3, 1, CV_64FC1);

  try {
    cv::solvePnP(object_points, image_points, camera_matrix_,
                 distortion_coefficients_, rvec, tvec, false,
                 cv::SOLVEPNP_SQPNP);
  } catch (const std::exception& e) {
    LOG(WARNING) << "Caught solvePnP exception: " << e.what();
    return std::nullopt;
  }

  cv::Mat field_to_camera = utils::MakeTransform(rvec, tvec).inv();
  cv::Mat field_to_robot = field_to_camera * camera_to_robot_;
  const int num_tags = static_cast<int>(tag_ids.size());

  position_estimate_t estimate;
  estimate.tag_ids = std::move(tag_ids);
  estimate.rejected_tag_ids = std::move(rejected_tag_ids);
  estimate.pose = utils::ConvertOpencvTransformationMatrixToWpilibPose(
      field_to_robot);
  estimate.variance =
      Variance(num_tags, avg_distance, kVarianceMin, kVarianceScalar);
  estimate.num_tags = num_tags;
  estimate.avg_tag_dist = avg_distance;

  return ambiguous_estimate_t{
      .pos1 = std::move(estimate),
      .pos2 = std::nullopt};
}

}  // namespace localization
