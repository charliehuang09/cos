#include "camera/uvc_camera_config.h"

#include <fstream>

#include "absl/log/check.h"

#include <nlohmann/json.hpp>

namespace camera {

UVCCameraConfig::UVCCameraConfig(const std::string& path) {
  std::ifstream file(path);
  CHECK(file.is_open());
  nlohmann::json config = nlohmann::json::parse(file);

  CHECK(config.at("camera_type").get<std::string>() == "uvc");
  name = config.at("name").get<std::string>();
  if (config.at("serial_id").is_null()) {
    serial_id = std::nullopt;
  } else {
    serial_id = config.at("serial_id").get<std::string>();
  }
  if (!config.at("exposure_time_ms").is_null()) {
    exposure_time_ms = config.at("exposure_time_ms").get<double>();
  }
  if (!config.at("auto_exposure").is_null()) {
    auto_exposure = config.at("auto_exposure").get<bool>();
  }
  height = config.at("height").get<int>();
  width = config.at("width").get<int>();
  fps = config.at("fps").get<int>();
  max_payload_size = config.at("max_payload_size").get<int>();
  max_frame_size = config.at("max_frame_size").get<int>();
}

}  // namespace camera
