#pragma once

#include <span>
#include <string>
#include <vector>

#include <wpi/struct/Struct.h>

#include "logging/log_type.h"
#include "logging/wpilog_writer.h"

namespace logging {

template <wpi::StructSerializable Message>
struct StructLog {
  static void Register(WPILogWriter& writer, const std::string& channel) {
    wpi::ForEachStructSchema<Message>(
        [&writer](std::string_view type, std::string_view schema) {
          writer.AddStructSchema(type, schema);
        });
    writer.RegisterRaw(channel, wpi::GetStructTypeString<Message>());
  }
  static void Write(WPILogWriter& writer, const std::string& channel,
                    const Message& message) {
    std::vector<uint8_t> buffer(wpi::GetStructSize<Message>());
    wpi::PackStruct(buffer, message);
    writer.WriteRaw(channel, buffer);
  }
};

template <typename Message, wpi::StructSerializable Element,
          std::vector<Element> Message::*Member>
struct StructArrayLog {
  static void Register(WPILogWriter& writer, const std::string& channel) {
    wpi::ForEachStructSchema<Element>(
        [&writer](std::string_view type, std::string_view schema) {
          writer.AddStructSchema(type, schema);
        });
    writer.RegisterRaw(
        channel,
        wpi::MakeStructArrayTypeString<Element, std::dynamic_extent>());
  }
  static void Write(WPILogWriter& writer, const std::string& channel,
                    const Message& message) {
    const auto& values = message.*Member;
    const size_t struct_size = wpi::GetStructSize<Element>();
    std::vector<uint8_t> buffer(values.size() * struct_size);
    auto output = std::span<uint8_t>{buffer};
    for (const auto& value : values) {
      wpi::PackStruct(output, value);
      output = output.subspan(struct_size);
    }
    writer.WriteRaw(channel, buffer);
  }
};

}  // namespace logging
