#include "utils/cv_geometry.h"

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

  const cv::Mat wpi_to_cv = EigenToCvMat(BasisMatrix(Basis::kWpiToCv));
  const cv::Mat expected =
      wpi_to_cv * EigenToCvMat(pose.ToMatrix()) * wpi_to_cv.t();

  EXPECT_LE(cv::norm(Pose3dToCvMat(pose) - expected), 1e-12);
}

TEST(CvGeometryTest, Transform3dToCvMatChangesTheEntireTransformBasis) {
  const frc::Transform3d transform(
      frc::Translation3d(units::meter_t{1.0}, units::meter_t{2.0},
                         units::meter_t{3.0}),
      frc::Rotation3d(units::radian_t{0.2}, units::radian_t{-0.3},
                      units::radian_t{0.4}));

  const cv::Mat wpi_to_cv = EigenToCvMat(BasisMatrix(Basis::kWpiToCv));
  const cv::Mat expected =
      wpi_to_cv * EigenToCvMat(transform.ToMatrix()) * wpi_to_cv.t();
  const cv::Mat actual = Transform3dToCvMat(transform);

  EXPECT_LE(cv::norm(actual - expected), 1e-12);
  EXPECT_DOUBLE_EQ(actual.at<double>(0, 3), -2.0);
  EXPECT_DOUBLE_EQ(actual.at<double>(1, 3), -3.0);
  EXPECT_DOUBLE_EQ(actual.at<double>(2, 3), 1.0);
}

}  // namespace
}  // namespace utils
