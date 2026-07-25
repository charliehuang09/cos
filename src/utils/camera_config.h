#pragma once

#include <nlohmann/json.hpp>
#include <opencv2/core/mat.hpp>
#include <frc/geometry/Transform3d.h>

namespace utils {

auto CameraMatrixFromJson(const nlohmann::json& intrinsics) -> cv::Mat;
auto DistortionCoefficientsFromJson(const nlohmann::json& intrinsics)
    -> cv::Mat;
auto ExtrinsicsJsonToCameraToRobot(const nlohmann::json& extrinsics)
    -> frc::Transform3d;

}  // namespace utils
