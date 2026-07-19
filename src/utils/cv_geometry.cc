#include "utils/cv_geometry.h"

#include <cmath>
#include <map>
#include <numbers>

#include <opencv2/calib3d.hpp>
#include <wpi/apriltag/AprilTagFieldLayout.hpp>
#include <wpi/math/geometry/Rotation3d.hpp>
#include <wpi/math/geometry/Translation3d.hpp>

namespace utils {

static const cv::Mat kCvToWpilib =
    (cv::Mat_<double>(4, 4) << 0, 0, 1, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0,
     1);
static const std::map<Basis, cv::Mat> kCvBases = {
    {Basis::kWpiToCv, kCvToWpilib.t()}, {Basis::kCvToWpi, kCvToWpilib}};

static auto ConvertOpencvCoordinateToWpilib(cv::Mat& vec) -> void {
  const double x = vec.ptr<double>()[2];
  const double y = vec.ptr<double>()[0];
  const double z = vec.ptr<double>()[1];
  vec.ptr<double>()[0] = x;
  vec.ptr<double>()[1] = -y;
  vec.ptr<double>()[2] = -z;
}

auto CvMatToPoint3d(const cv::Mat& mat) -> cv::Point3d {
  return {mat.at<double>(0), mat.at<double>(1), mat.at<double>(2)};
}

auto HomogenizePoint3d(cv::Point3d point) -> cv::Mat {
  return (cv::Mat_<double>(4, 1) << point.x, point.y, point.z, 1.0);
}

auto QuadAreaPixels(const std::array<cv::Point2d, 4>& corners) -> double {
  return 0.5 * std::abs((corners[0].x - corners[2].x) *
                            (corners[1].y - corners[3].y) -
                        (corners[1].x - corners[3].x) *
                            (corners[0].y - corners[2].y));
}

auto MakeTransform(const cv::Mat& rvec, const cv::Mat& tvec) -> cv::Mat {
  CV_Assert(rvec.total() == 3 && tvec.total() == 3);
  cv::Mat rotation;
  cv::Rodrigues(rvec, rotation);
  cv::Mat transform = cv::Mat::eye(4, 4, CV_64F);
  rotation.copyTo(transform(cv::Rect(0, 0, 3, 3)));
  transform.at<double>(0, 3) = tvec.at<double>(0);
  transform.at<double>(1, 3) = tvec.at<double>(1);
  transform.at<double>(2, 3) = tvec.at<double>(2);
  return transform;
}

auto EigenToCvMat(const Eigen::Matrix4d& mat) -> cv::Mat {
  cv::Mat cv_mat(mat.rows(), mat.cols(), CV_64F);
  Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                           Eigen::RowMajor>>(cv_mat.ptr<double>(),
                                             mat.rows(), mat.cols()) = mat;
  return cv_mat;
}

auto CvMatToEigen(const cv::Mat& mat) -> Eigen::Matrix4d {
  Eigen::Matrix4d out;
  for (int row = 0; row < 4; ++row) {
    for (int col = 0; col < 4; ++col) {
      out(row, col) = mat.at<double>(row, col);
    }
  }
  return out;
}

auto ChangeBasis(cv::Mat& mat, Basis basis) -> void {
  const cv::Mat& basis_mat = kCvBases.at(basis);
  mat = basis_mat * mat;
  if (mat.cols == mat.rows) {
    mat = mat * basis_mat.t();
  }
}

auto ConvertOpencvTransformationMatrixToWpilibPose(const cv::Mat& matrix)
    -> wpi::math::Pose3d {
  cv::Mat rotation = matrix(cv::Range(0, 3), cv::Range(0, 3)).clone();
  cv::Mat tvec = matrix(cv::Range(0, 3), cv::Range(3, 4)).clone();
  cv::Mat rvec;
  cv::Rodrigues(rotation, rvec);
  ConvertOpencvCoordinateToWpilib(tvec);
  ConvertOpencvCoordinateToWpilib(rvec);
  return wpi::math::Pose3d(CvMatToEigen(MakeTransform(rvec, tvec)));
}

auto ComputeRobotPose(const cv::Mat& tvec, const cv::Mat& rvec, int tag_id,
                      const wpi::apriltag::AprilTagFieldLayout& layout,
                      const cv::Mat& camera_to_robot) -> wpi::math::Pose3d {
  const cv::Mat camera_to_tag = MakeTransform(rvec, tvec);
  cv::Mat tag_to_camera = camera_to_tag.inv();
  ChangeBasis(tag_to_camera, Basis::kCvToWpi);

  cv::Mat rz_flip_wpi = (cv::Mat_<double>(3, 1) << 0, 0, std::numbers::pi);
  cv::Mat empty_tvec = (cv::Mat_<double>(3, 1) << 0, 0, 0);
  const cv::Mat rotate_yaw_wpilib = MakeTransform(rz_flip_wpi, empty_tvec);

  const cv::Mat field_to_tag =
      EigenToCvMat(layout.GetTagPose(tag_id).value().ToMatrix());
  const cv::Mat field_to_robot =
      field_to_tag * rotate_yaw_wpilib * tag_to_camera * camera_to_robot;
  return wpi::math::Pose3d{CvMatToEigen(field_to_robot)};
}

auto Pose3dToCvMat(wpi::math::Pose3d pose) -> cv::Mat {
  wpi::math::Pose3d opencv_pose(
      wpi::math::Translation3d(-pose.Y(), -pose.Z(), pose.X()),
      wpi::math::Rotation3d(-pose.Rotation().Y(), -pose.Rotation().Z(),
                            pose.Rotation().X()));
  return EigenToCvMat(opencv_pose.ToMatrix());
}

auto Transform3dToCvMat(wpi::math::Transform3d transform) -> cv::Mat {
  wpi::math::Pose3d opencv_pose(
      wpi::math::Translation3d(-transform.Y(), -transform.Z(), transform.X()),
      wpi::math::Rotation3d(-transform.Rotation().Y(),
                            -transform.Rotation().Z(),
                            transform.Rotation().X()));
  return EigenToCvMat(opencv_pose.ToMatrix());
}

}  // namespace utils
