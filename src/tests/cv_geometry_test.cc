#include "utils/cv_geometry.h"

#include <vector>

#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Rotation3d.h>
#include <frc/geometry/Transform3d.h>
#include <frc/geometry/Translation3d.h>
#include <gtest/gtest.h>

namespace utils {
namespace {

TEST(CvGeometryTest, Pose3dToCvMatChangesTheEntireTransformBasis) {
  const frc::Pose3d pose(
      frc::Translation3d(units::meter_t{1.0}, units::meter_t{2.0},
                         units::meter_t{3.0}),
      frc::Rotation3d(units::radian_t{0.2}, units::radian_t{-0.3},
                      units::radian_t{0.4}));

  const cv::Matx44d wpi_to_cv = EigenToCvMat(BasisMatrix(Basis::kWpiToCv));
  const cv::Matx44d expected =
      wpi_to_cv * EigenToCvMat(pose.ToMatrix()) * wpi_to_cv.t();

  EXPECT_LE(cv::norm(Pose3dToCvMat(pose) - expected), 1e-12);
}

TEST(CvGeometryTest, Transform3dToCvMatChangesTheEntireTransformBasis) {
  const frc::Transform3d transform(
      frc::Translation3d(units::meter_t{1.0}, units::meter_t{2.0},
                         units::meter_t{3.0}),
      frc::Rotation3d(units::radian_t{0.2}, units::radian_t{-0.3},
                      units::radian_t{0.4}));

  const cv::Matx44d wpi_to_cv = EigenToCvMat(BasisMatrix(Basis::kWpiToCv));
  const cv::Matx44d expected =
      wpi_to_cv * EigenToCvMat(transform.ToMatrix()) * wpi_to_cv.t();
  const cv::Matx44d actual = Transform3dToCvMat(transform);

  EXPECT_LE(cv::norm(actual - expected), 1e-12);
  EXPECT_DOUBLE_EQ(actual(0, 3), -2.0);
  EXPECT_DOUBLE_EQ(actual(1, 3), -3.0);
  EXPECT_DOUBLE_EQ(actual(2, 3), 1.0);
}

TEST(CvGeometryTest, CalibratesIntoFixedSizeIntrinsicsAndDistortion) {
  const cv::Matx33d intrinsics{800, 0, 320, 0, 810, 240, 0, 0, 1};
  const cv::Vec<double, 5> distortion{-0.1, 0.02, 0.001, -0.002, 0.0};
  std::vector<cv::Point3f> board;
  for (int row = 0; row < 6; ++row) {
    for (int col = 0; col < 8; ++col) {
      board.emplace_back(col * 0.04f, row * 0.04f, 0.0f);
    }
  }
  std::vector<std::vector<cv::Point3f>> object_points;
  std::vector<std::vector<cv::Point2f>> image_points;
  for (int view = 0; view < 10; ++view) {
    const cv::Vec3d rotation{0.04 * view - 0.2, 0.1 - 0.03 * view,
                            0.02 * view};
    const cv::Vec3d translation{-0.1 + 0.01 * view, -0.08,
                               0.7 + 0.03 * view};
    object_points.push_back(board);
    image_points.emplace_back();
    cv::projectPoints(board, rotation, translation, intrinsics, distortion,
                      image_points.back());
  }

  cv::Matx33d calibrated_intrinsics = cv::Matx33d::eye();
  cv::Vec<double, 5> calibrated_distortion{};
  const double error = cv::calibrateCamera(
      object_points, image_points, {640, 480}, calibrated_intrinsics,
      calibrated_distortion, cv::noArray(), cv::noArray(), cv::noArray(),
      cv::noArray(), cv::noArray());

  EXPECT_LT(error, 1e-3);
  EXPECT_NEAR(calibrated_intrinsics(0, 0), intrinsics(0, 0), 0.1);
  EXPECT_NEAR(calibrated_intrinsics(1, 1), intrinsics(1, 1), 0.1);
  EXPECT_NEAR(calibrated_distortion[0], distortion[0], 1e-3);
}

}  // namespace
}  // namespace utils
