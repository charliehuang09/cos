#pragma once
#include <nadjieb/mjpeg_streamer.hpp>
#include "camera/uvc_camera_node.h"

namespace streamer {

class JpegBufferStreamerNode {
 public:
  JpegBufferStreamerNode(std::string_view input_path, std::string path,
                         int port);
  auto CreateCallback() -> std::function<void(const control_loop::Context&)>;

 private:
  void Stream(const camera::JpegBuffer& jpeg_buffer);

 private:
  nadjieb::MJPEGStreamer streamer_;
  std::string input_path_;
  std::string path_;
};

}  // namespace streamer
