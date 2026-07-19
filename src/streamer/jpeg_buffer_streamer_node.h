#pragma once
#include <string>

#include <nadjieb/mjpeg_streamer.hpp>

#include "camera/uvc_camera_node.h"

namespace streamer {

class JpegBufferStreamerNode final {
 public:
  JpegBufferStreamerNode(std::string path, int port);
  void Stream(const camera::JpegBuffer& jpeg_buffer);

 private:
  nadjieb::MJPEGStreamer streamer_;
  std::string path_;
};

}  // namespace streamer
