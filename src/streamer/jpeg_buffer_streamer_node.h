#pragma once
#include <nadjieb/mjpeg_streamer.hpp>

#include "camera/uvc_camera_node.h"

namespace streamer {

class JpegBufferStreamerNode final : public control_loop::INode {
 public:
  JpegBufferStreamerNode(std::string_view input_path, std::string path,
                         int port);
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  void Stream(const camera::JpegBuffer& jpeg_buffer);

 private:
  nadjieb::MJPEGStreamer streamer_;
  std::string input_path_;
  std::string path_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

}  // namespace streamer
