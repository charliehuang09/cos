#include "streamer/jpeg_buffer_streamer_node.h"

namespace streamer {

JpegBufferStreamerNode::JpegBufferStreamerNode(std::string_view input_path,
                                               std::string path, int port)
    : input_path_(input_path),
      path_(std::move(path)),
      dependencies_({input_path_}) {
  streamer_.start(port);
}

void JpegBufferStreamerNode::Stream(const camera::JpegBuffer& jpeg_buffer) {
  std::string string_buffer(static_cast<char*>(jpeg_buffer.ptr),
                            jpeg_buffer.size);
  streamer_.publish(path_, string_buffer);
}

auto JpegBufferStreamerNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    auto* jpeg_buffer = context->GetMessage<camera::JpegBuffer>(input_path_);
    if (jpeg_buffer == nullptr || jpeg_buffer->ptr == nullptr) {
      return;
    }
    Stream(*jpeg_buffer);
  };
}

auto JpegBufferStreamerNode::GetDependencies()
    -> const std::vector<std::string>& {
  return dependencies_;
}

}  // namespace streamer
