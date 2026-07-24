#include <fstream>
#include <iostream>

#include "camera/uvc_camera_node.h"
#include "logging/jpeg_buffer_log_node.h"

namespace logging {

JpegBufferLogNode::JpegBufferLogNode(std::string_view input_channel,
                                     std::string_view folder_path,
                                     control_loop::ThreadPool& thread_pool)
    : input_channel_(input_channel),
      folder_path_(folder_path),
      dependencies_({{input_channel_, typeid(camera::JpegBuffer)}}),
      thread_pool_(thread_pool) {}

auto JpegBufferLogNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    thread_pool_.Submit([this, context]() -> void {
      auto jpeg_buffer =
          context->GetMessage<camera::JpegBuffer>(input_channel_);
      if (jpeg_buffer != nullptr) {
        std::string output_file_path = folder_path_ + "/" +
                                       std::to_string(jpeg_buffer->timestamp) +
                                       ".jpeg_buffer";
        std::ofstream out(output_file_path, std::ios::binary);
        out.write(reinterpret_cast<const char*>(jpeg_buffer->ptr),
                  jpeg_buffer->size);
      }
    });
  };
}

auto JpegBufferLogNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto JpegBufferLogNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void JpegBufferLogNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}

}  // namespace logging
