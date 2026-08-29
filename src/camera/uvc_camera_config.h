#pragma once

#include <optional>
#include <string>

namespace camera {

struct UVCCameraConfig {
  UVCCameraConfig(const std::string& path);

  std::string name;                      // For debugging
  std::optional<std::string> serial_id;  // Used to find which camera to use
  int height;
  int width;
  int fps;
  double exposure_time_ms = 15.7;
  bool auto_exposure = true;
  int max_payload_size = 3072;
  int max_frame_size = 2048589;
};

// Default UVC stream control values:
// bmHint: 0001
// bFormatIndex: 1
// bFrameIndex: 1
// dwFrameInterval: 83333
// wKeyFrameRate: 0
// wPFrameRate: 0
// wCompQuality: 0
// wCompWindowSize: 0
// wDelay: 0
// dwMaxVideoFrameSize: 2048589
// dwMaxPayloadTransferSize: 3072
// bInterfaceNumber: 1

}  // namespace camera
