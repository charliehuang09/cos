#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <wpi/math/geometry/Transform3d.hpp>

namespace utils {

auto CameraMatrixFromJson(const nlohmann::json& intrinsics) -> cv::Mat;
auto DistortionCoefficientsFromJson(const nlohmann::json& intrinsics)
    -> cv::Mat;
auto ExtrinsicsJsonToCameraToRobot(const nlohmann::json& extrinsics)
    -> wpi::math::Transform3d;

}  // namespace utils
