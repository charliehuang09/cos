#pragma once

#include <span>
#include <string_view>
#include <vector>

#include "control_loop/message.h"

namespace logging {

// Message serializers are free function specializations declared near each
// serializable message. Unserializable publications fail at compile time.
template <typename T>
std::vector<int> StartLog(WPILogWriter& writer, std::string_view channel) = delete;

template <typename T>
void AppendLog(WPILogWriter& writer, std::span<const int> entries, const T& message,
               int64_t timestamp) = delete;

}  // namespace logging

namespace control_loop {

template <typename T>
auto MessageDescriptor::For(std::string_view channel) -> MessageDescriptor {
  if constexpr (!T::loggable) {
    return {channel, typeid(T)};
  } else {
    return {channel, typeid(T),
            [](logging::WPILogWriter& writer, std::string_view name) {
              return logging::StartLog<T>(writer, name);
            },
            [](logging::WPILogWriter& writer, std::span<const int> entries,
               const IMessage& message, int64_t timestamp) {
              logging::AppendLog<T>(writer, entries,
                                    static_cast<const T&>(message), timestamp);
            }};
  }
}

}  // namespace control_loop
