#pragma once

#include <array>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>

#include <wpi/DataLogWriter.h>
#include <wpi/struct/Struct.h>
#include <wpi/timestamp.h>

#include "control_loop/message.h"
#include "logging/log_type.h"

namespace control_loop {
class ControlLoop;
struct ContextInternal;
}  // namespace control_loop

namespace logging {

// A terminal node that records the messages accumulated in a context.
// Messages over max_message_bytes are omitted.
class WPILogWriter final {
 public:
  explicit WPILogWriter(const control_loop::ControlLoop& control_loop);
  ~WPILogWriter();

  template <typename T>
  void RegisterLoggable(const std::string& channel);

  template <typename T>
  void Write(const std::string& channel, const T& value);

  void RegisterRaw(std::string_view channel, std::string_view type);
  void WriteRaw(std::string_view channel, std::span<const uint8_t> value);
  void AddStructSchema(std::string_view type, std::string_view schema);

  void Write(const control_loop::ContextInternal& context);

  // Flushes all buffered records and closes the WPILOG file. No records are
  // accepted after this returns.
  void Close();

 private:
  void RegisterPublication(const control_loop::MessageDescriptor& publication);

  std::unique_ptr<wpi::log::DataLogWriter> writer_;
  std::unordered_map<std::string, int> entry_indices_;
  std::unordered_map<std::string, WriteLogTypeFunction> write_functions_;
  bool closed_ = false;
};

template <typename T>
void WPILogWriter::RegisterLoggable(const std::string& channel) {
  int entry_index;
  if constexpr (std::is_same_v<T, bool>) {
    entry_index = writer_->Start(channel, "boolean");
  } else if constexpr (std::is_integral_v<T>) {
    entry_index = writer_->Start(channel, "int64");
  } else if constexpr (std::is_floating_point_v<T>) {
    entry_index = writer_->Start(channel, "double");
  } else if constexpr (std::is_same_v<T, std::string> ||
                       std::is_same_v<T, std::string_view>) {
    entry_index = writer_->Start(channel, "string");
  } else if constexpr (wpi::StructSerializable<T>) {
    writer_->AddStructSchema<T>();
    entry_index = writer_->Start(channel, wpi::GetStructTypeString<T>());
  } else {
    static_assert(std::is_same_v<T, void>,
                  "Unsupported primitive type for WPILogWriter");
  }
  entry_indices_[channel] = entry_index;
}

template <typename T>
void WPILogWriter::Write(const std::string& channel, const T& value) {
  const int entry = entry_indices_.at(channel);
  const int64_t timestamp = wpi::Now();

  if constexpr (std::is_same_v<T, bool>) {
    writer_->AppendBoolean(entry, value, timestamp);
  } else if constexpr (std::is_integral_v<T>) {
    writer_->AppendInteger(entry, static_cast<int64_t>(value), timestamp);
  } else if constexpr (std::is_floating_point_v<T>) {
    writer_->AppendDouble(entry, static_cast<double>(value), timestamp);
  } else if constexpr (std::is_same_v<T, std::string> ||
                       std::is_same_v<T, std::string_view>) {
    writer_->AppendString(entry, value, timestamp);
  } else if constexpr (wpi::StructSerializable<T>) {
    std::array<uint8_t, wpi::GetStructSize<T>()> buffer;
    wpi::Struct<T>::Pack(buffer, value);
    writer_->AppendRaw(entry, buffer, timestamp);
  } else {
    static_assert(std::is_same_v<T, void>,
                  "Unsupported type for WPILogWriter");
  }
}

}  // namespace logging
