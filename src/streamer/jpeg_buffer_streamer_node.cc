#include "streamer/jpeg_buffer_streamer_node.h"

#include <utility>

namespace streamer {

JpegBufferStreamerNode::JpegBufferStreamerNode(std::string_view input_path,
                                               std::string path, int port)
    : input_path_(input_path),
      path_(std::move(path)) {
  streamer_.start(port);
}

void JpegBufferStreamerNode::Stream(const camera::JpegBuffer& jpeg_buffer) {
  std::string string_buffer(static_cast<char*>(jpeg_buffer.ptr),
                            jpeg_buffer.size);
  streamer_.publish(path_, string_buffer);
}

void JpegBufferStreamerNode::RegisterCallback(
    const control_loop::Context& context) {
  auto* jpeg_buffer = context->GetMessage<camera::JpegBuffer>(input_path_);
  if (jpeg_buffer == nullptr || jpeg_buffer->ptr == nullptr) {
    return;
  }
  Stream(*jpeg_buffer);
}

}  // namespace streamer
