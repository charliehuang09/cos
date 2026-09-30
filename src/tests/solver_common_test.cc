#include "localization/solver_common.h"

#include <numbers>
#include <limits>

#include <frc/geometry/Pose3d.h>
#include <frc/geometry/Rotation3d.h>
#include <gtest/gtest.h>

namespace {

auto Pose(double x, double y, double z, double roll = 0.0,
          double pitch = 0.0, double yaw = 0.0) -> frc::Pose3d {
  return frc::Pose3d{
      units::meter_t{x}, units::meter_t{y}, units::meter_t{z},
      frc::Rotation3d{units::radian_t{roll}, units::radian_t{pitch},
                      units::radian_t{yaw}}};
}

TEST(SolverCommonTest, AcceptsGroundRobotPose) {
  EXPECT_FALSE(localization::PoseOffField(Pose(8.0, 4.0, 0.0)));
  EXPECT_FALSE(localization::PoseOffField(
      Pose(8.0, 4.0, 0.19, std::numbers::pi / 12.1,
           -std::numbers::pi / 12.1, std::numbers::pi)));
}

TEST(SolverCommonTest, RejectsInvalidHeight) {
  EXPECT_TRUE(localization::PoseOffField(Pose(8.0, 4.0, 0.21)));
  EXPECT_TRUE(localization::PoseOffField(Pose(8.0, 4.0, -0.21)));
}

TEST(SolverCommonTest, RejectsInvalidRollOrPitch) {
  constexpr double kInvalidTilt = std::numbers::pi / 12.0 + 0.01;
  EXPECT_TRUE(localization::PoseOffField(Pose(8.0, 4.0, 0.0, kInvalidTilt)));
  EXPECT_TRUE(localization::PoseOffField(Pose(8.0, 4.0, 0.0, -kInvalidTilt)));
  EXPECT_TRUE(
      localization::PoseOffField(Pose(8.0, 4.0, 0.0, 0.0, kInvalidTilt)));
  EXPECT_TRUE(
      localization::PoseOffField(Pose(8.0, 4.0, 0.0, 0.0, -kInvalidTilt)));
}

TEST(SolverCommonTest, StillRejectsPositionsOutsideField) {
  EXPECT_TRUE(localization::PoseOffField(Pose(-0.21, 4.0, 0.0)));
  EXPECT_TRUE(localization::PoseOffField(Pose(16.75, 4.0, 0.0)));
  EXPECT_TRUE(localization::PoseOffField(Pose(8.0, -0.21, 0.0)));
  EXPECT_TRUE(localization::PoseOffField(Pose(8.0, 8.21, 0.0)));
}

TEST(SolverCommonTest, RejectsNonFinitePose) {
  EXPECT_TRUE(localization::PoseOffField(Pose(
      8.0, 4.0, std::numeric_limits<double>::quiet_NaN())));
}

}  // namespace
