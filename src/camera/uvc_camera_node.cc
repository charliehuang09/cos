#include "absl/log/check.h"
#include "absl/log/log.h"

#include "camera/uvc_camera_node.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <wpi/system/Timer.hpp>

namespace camera {

UVCCameraConfig::UVCCameraConfig(const std::string& path) {
  std::ifstream file(path);
  CHECK(file.is_open());
  nlohmann::json config = nlohmann::json::parse(file);

  CHECK(config.at("camera_type").get<std::string>() == "uvc");
  name = config.at("name").get<std::string>();
  if (config.at("serial_id").is_null()) {
    serial_id = std::nullopt;
  } else {
    serial_id = config.at("serial_id").get<std::string>();
  }
  height = config.at("height").get<int>();
  width = config.at("width").get<int>();
  fps = config.at("fps").get<int>();
  max_payload_size = config.at("max_payload_size").get<int>();
  max_frame_size = config.at("max_frame_size").get<int>();
}

UVCCameraNode::UVCCameraNode(std::string_view output_path,
                             const UVCCameraConfig& config)
    : output_path_(output_path),
      name_(config.name),
      publications_({{output_path_, typeid(JpegBuffer)}}) {
  {
    uvc_error_t code = uvc_init(&context_, nullptr);
    CHECK(!code) << "UVC failed to init will error code: " << code;
  }
  {
    const char* serial_id =
        config.serial_id.has_value() ? config.serial_id->c_str() : nullptr;
    uvc_error_t code = uvc_find_device(context_, &device_, 0, 0, serial_id);
    CHECK(!code) << "UVC failed to find device with error code: " << code
                 << " camera_name: " << config.name;
  }
  {
    uvc_error_t code = uvc_open(device_, &device_handle_);
    CHECK(!code) << "UVC failed to open device with error code: " << code
                 << " camera name: " << config.name;
  }
  {
    uvc_error_t code = uvc_get_stream_ctrl_format_size(
        device_handle_, &ctrl_, UVC_FRAME_FORMAT_MJPEG, config.width,
        config.height, config.fps);
    CHECK(!code) << "UVC failed to get stream ctrl format with exit code: "
                 << code << " camera_name: " << config.name;

    ctrl_.dwMaxPayloadTransferSize = config.max_payload_size;
    ctrl_.dwMaxVideoFrameSize = config.max_frame_size;
  }
}

auto UVCCameraNode::CreateCallback()
    -> std::function<void(const control_loop::Context&)> {
  return [this](const control_loop::Context& context) -> void {
    Callback(context);
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  };
}

void UVCCameraNode::CallBack(uvc_frame_t* frame) {
  CHECK(frame->frame_format == UVC_COLOR_FORMAT_MJPEG);
  auto buffer = std::make_unique<JpegBuffer>(
      frame->data_bytes, wpi::Timer::GetMonotonicTimestamp().value());
  std::memcpy(buffer->ptr, frame->data, frame->data_bytes);

  {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_ = std::move(buffer);
  }
}

void UVCCameraNode::Callback(const control_loop::Context& context) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (buffer_ == nullptr) {
    return;
  }
  context->SetMessage(output_path_, std::move(buffer_));
}

void UVCCameraNode::Start() {
  int code = uvc_start_streaming(
      device_handle_, &ctrl_,
      [](uvc_frame_t* frame, void* ptr) -> void {
        auto uvc_camera_node = static_cast<UVCCameraNode*>(ptr);
        uvc_camera_node->CallBack(frame);
      },
      this, 0);
  CHECK(!code) << "UVC failed to start streaming with exit code: " << code
               << " camera name: " << name_;
}

UVCCameraNode::~UVCCameraNode() {
  uvc_stop_streaming(device_handle_);
  uvc_close(device_handle_);
  uvc_unref_device(device_);
  uvc_exit(context_);
  LOG(INFO) << name_ << " has been destructed";
}

auto UVCCameraNode::GetDependencies() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return dependencies_;
}

auto UVCCameraNode::GetPublications() const
    -> const std::vector<control_loop::MessageDescriptor>& {
  return publications_;
}

void UVCCameraNode::RegisterCallback(
    const std::function<void(const control_loop::Context&)>& callback) {
  callbacks_.emplace_back(callback);
}
}  // namespace camera
