#include "streamer/jpeg_buffer_streamer_node.h"

namespace streamer {
JpegBufferStreamerNode::JpegBufferStreamerNode(std::string path, int port)
    : path_(std::move(path)) {
  streamer_.start(port);
}

void JpegBufferStreamerNode::Stream(
    const std::shared_ptr<camera::JpegBuffer>& jpeg_buffer) {
  std::string string_buffer(static_cast<char*>(jpeg_buffer->ptr()),
                            jpeg_buffer->size());
  streamer_.publish(path_, string_buffer);
}

}  // namespace streamer
