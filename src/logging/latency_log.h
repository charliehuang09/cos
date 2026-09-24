#pragma once

#include "control_loop/timed_node.h"
#include "logging/publication.h"

namespace logging {

template <>
std::vector<int> StartLog<control_loop::LatencyMessage>(WPILogWriter& writer,
                                            std::string_view channel);

template <>
void AppendLog<control_loop::LatencyMessage>(
    WPILogWriter& writer, std::span<const int> entries,
    const control_loop::LatencyMessage& message, int64_t timestamp);

}  // namespace logging
