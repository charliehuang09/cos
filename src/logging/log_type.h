#pragma once

#include <string>
#include <string_view>
#include <typeindex>
#include <unordered_map>

#include "control_loop/message.h"

namespace logging {

class WPILogWriter;

// Marks a published message as intentionally excluded from the log.
struct NoLog {};

// Specialized alongside message types that have a log representation.
template <typename Message>
struct LogType {
  static void Register(WPILogWriter&, const std::string&) {}
  static void Write(WPILogWriter&, const std::string&, const Message&) {}
};

using RegisterLogTypeFunction = void (*)(WPILogWriter&, std::string_view);
using WriteLogTypeFunction = void (*)(WPILogWriter&, std::string_view,
                                      const control_loop::IMessage&);

struct LogTypeHandler {
  RegisterLogTypeFunction register_log_type;
  WriteLogTypeFunction write_log_type;
};

auto GetLogTypeRegistry()
    -> std::unordered_map<std::type_index, LogTypeHandler>&;

template <typename Message>
void RegisterLogType() {
  GetLogTypeRegistry()[typeid(Message)] = {
      [](WPILogWriter& writer, std::string_view channel) {
        LogType<Message>::Register(writer, std::string(channel));
      },
      [](WPILogWriter& writer, std::string_view channel,
         const control_loop::IMessage& message) {
        const auto* typed_message = dynamic_cast<const Message*>(&message);
        if (typed_message != nullptr) {
          LogType<Message>::Write(writer, std::string(channel), *typed_message);
        }
      }};
}

void RegisterBuiltInLogTypes();

}  // namespace logging
