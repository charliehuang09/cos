#include "logging/latency_log.h"

#include "logging/wpilog_writer.h"

namespace logging {

template <>
std::vector<int> StartLog<control_loop::LatencyMessage>(WPILogWriter& writer,
                                            std::string_view channel) {
  return {writer.StartDouble(channel)};
}

template <>
void AppendLog<control_loop::LatencyMessage>(
    WPILogWriter& writer, std::span<const int> entries,
    const control_loop::LatencyMessage& message, int64_t timestamp) {
  writer.AppendDouble(entries[0], message.latency.count(), timestamp);
}

}  // namespace logging
