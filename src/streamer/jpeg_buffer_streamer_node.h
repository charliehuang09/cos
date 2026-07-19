#pragma once
#include <functional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <nadjieb/mjpeg_streamer.hpp>

#include "camera/uvc_camera_node.h"

namespace streamer {

class JpegBufferStreamerNode final : public control_loop::INode {
 public:
  JpegBufferStreamerNode(std::string_view input_path, std::string path,
                         int port);
  void RegisterCallback(
      std::function<void(const control_loop::Context&)> callback) override {
    callbacks_.emplace_back(std::move(callback));
  }
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;

 private:
  void Stream(const camera::JpegBuffer& jpeg_buffer);

 private:
  nadjieb::MJPEGStreamer streamer_;
  std::string input_path_;
  std::string path_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
};

}  // namespace streamer
