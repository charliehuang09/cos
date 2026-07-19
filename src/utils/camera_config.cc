#include "utils/camera_config.h"

#include <wpi/math/geometry/Pose3d.hpp>
#include <wpi/math/geometry/Rotation3d.hpp>

namespace utils {

auto CameraMatrixFromJson(const nlohmann::json& intrinsics) -> cv::Mat {
  return (cv::Mat_<double>(3, 3) << intrinsics.at("fx").get<double>(), 0.0,
          intrinsics.at("cx").get<double>(), 0.0,
          intrinsics.at("fy").get<double>(), intrinsics.at("cy").get<double>(),
          0.0, 0.0, 1.0);
}

auto DistortionCoefficientsFromJson(const nlohmann::json& intrinsics)
    -> cv::Mat {
  return (cv::Mat_<double>(1, 5) << intrinsics.at("k1").get<double>(),
          intrinsics.at("k2").get<double>(), intrinsics.at("p1").get<double>(),
          intrinsics.at("p2").get<double>(), intrinsics.at("k3").get<double>());
}

auto ExtrinsicsJsonToCameraToRobot(const nlohmann::json& extrinsics)
    -> wpi::math::Transform3d {
  wpi::math::Pose3d camera_pose(
      wpi::units::meter_t{extrinsics.at("translation_x").get<double>()},
      wpi::units::meter_t{extrinsics.at("translation_y").get<double>()},
      wpi::units::meter_t{extrinsics.at("translation_z").get<double>()},
      wpi::math::Rotation3d(
          wpi::units::radian_t{extrinsics.at("rotation_x").get<double>()},
          wpi::units::radian_t{extrinsics.at("rotation_y").get<double>()},
          wpi::units::radian_t{extrinsics.at("rotation_z").get<double>()}));
  wpi::math::Transform3d robot_to_camera(wpi::math::Pose3d{}, camera_pose);
  return robot_to_camera.Inverse();
}

}  // namespace utils
