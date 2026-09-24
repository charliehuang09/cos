#include "logging/tag_detections_log.h"

#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

#include <wpi/struct/Struct.h>

#include "logging/wpilog_writer.h"

namespace logging {
namespace {

constexpr std::string_view kType = "struct:TagDetection";
constexpr std::string_view kArrayType = "struct:TagDetection[]";
constexpr std::string_view kSchema =
    "int32 tag_id;double corner_x[4];double corner_y[4]";
constexpr size_t kDetectionBytes =
    wpi::GetStructSize<int32_t>() + 8 * wpi::GetStructSize<double>();

}  // namespace

template <>
std::vector<int> StartLog<apriltag::TagDetections>(WPILogWriter& writer,
                                      std::string_view channel) {
  writer.AddStructSchema(kType, kSchema);
  return {writer.StartRaw(channel, kArrayType)};
}

template <>
void AppendLog<apriltag::TagDetections>(
    WPILogWriter& writer, std::span<const int> entries, const apriltag::TagDetections& message,
    int64_t timestamp) {
  std::vector<uint8_t> buffer(message.tag_detections.size() *
                              kDetectionBytes);
  auto output = std::span<uint8_t>{buffer};
  for (const auto& detection : message.tag_detections) {
    wpi::PackStruct(output, static_cast<int32_t>(detection.tag_id));
    output = output.subspan(wpi::GetStructSize<int32_t>());
    for (const auto& corner : detection.corners) {
      wpi::PackStruct(output, corner.x);
      output = output.subspan(wpi::GetStructSize<double>());
    }
    for (const auto& corner : detection.corners) {
      wpi::PackStruct(output, corner.y);
      output = output.subspan(wpi::GetStructSize<double>());
    }
  }
  writer.AppendRaw(entries[0], buffer, timestamp);
}

}  // namespace logging
