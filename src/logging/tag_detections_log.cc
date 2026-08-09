#include "apriltag/tag_detections.h"

#include <cstdint>
#include <span>
#include <string_view>

#include <wpi/struct/Struct.h>

#include "logging/struct_log.h"

template <>
struct wpi::Struct<apriltag::TagDetections::tag_detection> {
  using Detection = apriltag::TagDetections::tag_detection;

  static constexpr std::string_view GetTypeName() { return "TagDetection"; }
  static constexpr size_t GetSize() {
    return wpi::GetStructSize<int32_t>() +
           8 * wpi::GetStructSize<double>();
  }
  static constexpr std::string_view GetSchema() {
    return "int32 tag_id;double corner_x[4];double corner_y[4]";
  }

  static auto Unpack(std::span<const uint8_t> data) -> Detection {
    Detection detection;
    detection.tag_id = wpi::UnpackStruct<int32_t>(data);
    size_t offset = wpi::GetStructSize<int32_t>();
    for (auto& corner : detection.corners) {
      corner.x = wpi::UnpackStruct<double>(data.subspan(offset));
      offset += wpi::GetStructSize<double>();
    }
    for (auto& corner : detection.corners) {
      corner.y = wpi::UnpackStruct<double>(data.subspan(offset));
      offset += wpi::GetStructSize<double>();
    }
    return detection;
  }

  static void Pack(std::span<uint8_t> data, const Detection& detection) {
    wpi::PackStruct(data, static_cast<int32_t>(detection.tag_id));
    size_t offset = wpi::GetStructSize<int32_t>();
    for (const auto& corner : detection.corners) {
      wpi::PackStruct(data.subspan(offset), corner.x);
      offset += wpi::GetStructSize<double>();
    }
    for (const auto& corner : detection.corners) {
      wpi::PackStruct(data.subspan(offset), corner.y);
      offset += wpi::GetStructSize<double>();
    }
  }
};

static_assert(
    wpi::StructSerializable<apriltag::TagDetections::tag_detection>);

namespace logging {

template <>
struct LogType<apriltag::TagDetections> {
  static void Register(WPILogWriter& writer, const std::string& channel);
  static void Write(WPILogWriter& writer, const std::string& channel,
                    const apriltag::TagDetections& message);
};

using TagDetectionsLog =
    StructArrayLog<apriltag::TagDetections,
                   apriltag::TagDetections::tag_detection,
                   &apriltag::TagDetections::tag_detections>;

void LogType<apriltag::TagDetections>::Register(
    WPILogWriter& writer, const std::string& channel) {
  TagDetectionsLog::Register(writer, channel);
}

void LogType<apriltag::TagDetections>::Write(
    WPILogWriter& writer, const std::string& channel,
    const apriltag::TagDetections& message) {
  TagDetectionsLog::Write(writer, channel, message);
}

void RegisterBuiltInLogTypes() {
  RegisterLogType<apriltag::TagDetections>();
}

}  // namespace logging
