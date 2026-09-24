#pragma once

#include "apriltag/tag_detections.h"
#include "logging/publication.h"

namespace logging {

template <>
std::vector<int> StartLog<apriltag::TagDetections>(WPILogWriter& writer,
                                      std::string_view channel);

template <>
void AppendLog<apriltag::TagDetections>(WPILogWriter& writer, std::span<const int> entries,
                                        const apriltag::TagDetections& message,
                                        int64_t timestamp);

}  // namespace logging
