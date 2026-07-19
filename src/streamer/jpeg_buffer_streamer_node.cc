#include "streamer/jpeg_buffer_streamer_node.h"

#include "absl/log/check.h"

namespace streamer {
JpegBufferStreamerNode::JpegBufferStreamerNode(std::string_view input_path,
                                               std::string path, int port)
    : input_path_(input_path), path_(std::move(path)) {
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
    camera::JpegBuffer* jpeg_buffer =
        context->GetMessage<camera::JpegBuffer>(input_path_);
    if (jpeg_buffer == nullptr) {
      return;
    }
    Stream(*jpeg_buffer);
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

}  // namespace streamer
