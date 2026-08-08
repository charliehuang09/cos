#include "logging/wpilog_writer.h"

#include <string>
#include <system_error>
#include <type_traits>

#include "absl/log/log.h"
#include "wpi/DataLogWriter.h"

namespace logging {

WPILogWriter::WPILogWriter() {
  std::error_code ec;
  writer_ = std::make_unique<wpi::log::DataLogWriter>("output_file.wpilog",
                                                       ec);
  if (ec.value() != 0) {
    LOG(FATAL) << "Failed to initialize WPILog";
  }
}

template <typename T>
void WPILogWriter::RegisterLoggable(
    const std::string& channel) {
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
    entry_index = writer_->Start(
        channel,
        wpi::GetStructTypeString<T>());
  }
  else {
    static_assert(
        std::is_same_v<T, void>,
        "Unsupported primitive type for WPILogWriter");
  }
  entry_indices_[channel] = entry_index;
}

template <typename T>
void WPILogWriter::Write(
    const std::string& channel,
    const T& value) {

  const int entry = entry_indices_.at(std::string{channel});

  if constexpr (std::is_same_v<T, bool>) {
    writer_->AppendBoolean(entry, value);

  } else if constexpr (std::is_integral_v<T>) {
    writer_->AppendInteger(entry, static_cast<int64_t>(value));

  } else if constexpr (std::is_floating_point_v<T>) {
    writer_->AppendDouble(entry, static_cast<double>(value));

  } else if constexpr (
      std::is_same_v<T, std::string> ||
      std::is_same_v<T, std::string_view>) {
    writer_->AppendString(entry, value);

  } else if constexpr (wpi::StructSerializable<T>) {
    std::array<uint8_t, wpi::GetStructSize<T>()> buffer;
    wpi::Struct<T>::Pack(buffer, value);
    writer_->AppendRaw(entry, buffer);

  } else {
    static_assert(
        std::is_same_v<T, void>,
        "Unsupported type for WPILogWriter");
  }
}
}  // namespace logging
