#pragma once
#include <string>
#include <string_view>

#include <nadjieb/mjpeg_streamer.hpp>

#include "camera/uvc_camera_node.h"
#include "control_loop/context.h"

namespace streamer {

class JpegBufferStreamerNode final {
 public:
  JpegBufferStreamerNode(std::string_view input_path, std::string path,
                         int port);
  void RegisterCallback(const control_loop::Context& context);
  void Stream(const camera::JpegBuffer& jpeg_buffer);

 private:
  nadjieb::MJPEGStreamer streamer_;
  std::string input_path_;
  std::string path_;
};

}  // namespace streamer
