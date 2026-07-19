#pragma once

#include <Eigen/Core>
#include <array>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <wpi/math/geometry/Pose3d.hpp>
#include <wpi/math/geometry/Transform3d.hpp>

namespace wpi::apriltag {
class AprilTagFieldLayout;
}

namespace utils {

enum class Basis { kWpiToCv, kCvToWpi };

auto CvMatToPoint3d(const cv::Mat& mat) -> cv::Point3d;
auto HomogenizePoint3d(cv::Point3d point) -> cv::Mat;
auto QuadAreaPixels(const std::array<cv::Point2d, 4>& corners) -> double;
auto MakeTransform(const cv::Mat& rvec, const cv::Mat& tvec) -> cv::Mat;
auto EigenToCvMat(const Eigen::Matrix4d& mat) -> cv::Mat;
auto CvMatToEigen(const cv::Mat& mat) -> Eigen::Matrix4d;
auto ChangeBasis(cv::Mat& mat, Basis basis) -> void;
auto ConvertOpencvTransformationMatrixToWpilibPose(const cv::Mat& matrix)
    -> wpi::math::Pose3d;
auto ComputeRobotPose(const cv::Mat& tvec, const cv::Mat& rvec, int tag_id,
                      const wpi::apriltag::AprilTagFieldLayout& layout,
                      const cv::Mat& camera_to_robot) -> wpi::math::Pose3d;
auto Pose3dToCvMat(wpi::math::Pose3d pose) -> cv::Mat;
auto Transform3dToCvMat(wpi::math::Transform3d transform) -> cv::Mat;

}  // namespace utils
