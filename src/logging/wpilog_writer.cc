#include "logging/wpilog_writer.h"

#include <mutex>
#include <system_error>

#include "absl/log/log.h"
#include "control_loop/context.h"
#include "control_loop/control_loop.h"
#include "logging/log_type.h"
#include "wpi/DataLogWriter.h"
#include "wpi/timestamp.h"

namespace logging {

auto GetLogTypeRegistry()
    -> std::unordered_map<std::type_index, LogTypeHandler>& {
  static std::unordered_map<std::type_index, LogTypeHandler> registry;
  return registry;
}

WPILogWriter::WPILogWriter(const control_loop::ControlLoop& control_loop) {
  std::error_code ec;
  writer_ = std::make_unique<wpi::log::DataLogWriter>("output_file.wpilog",
                                                       ec);
  if (ec.value() != 0) {
    LOG(FATAL) << "Failed to initialize WPILog";
  }
  RegisterBuiltInLogTypes();
  for (const auto& publication : control_loop.GetPublications()) {
    RegisterPublication(publication);
  }
}

WPILogWriter::~WPILogWriter() {
  Close();
}

void WPILogWriter::RegisterPublication(
    const control_loop::MessageDescriptor& publication) {
  if (publication.GetTypes().size() != 1) {
    LOG(WARNING) << "Skipping WPILOG publication with multiple types on "
                 << publication.GetChannel();
    return;
  }

  const auto type = *publication.GetTypes().begin();
  const auto handler = GetLogTypeRegistry().find(type);
  if (handler == GetLogTypeRegistry().end()) {
    return;
  }

  const std::string& channel = publication.GetChannel();
  handler->second.register_log_type(*this, channel);
  write_functions_[channel] = handler->second.write_log_type;
}

void WPILogWriter::RegisterRaw(std::string_view channel,
                               std::string_view type) {
  entry_indices_[std::string(channel)] = writer_->Start(channel, type);
}

void WPILogWriter::WriteRaw(std::string_view channel,
                            std::span<const uint8_t> value) {
  const int entry = entry_indices_.at(std::string(channel));
  writer_->AppendRaw(entry, value, wpi::Now());
}

void WPILogWriter::AddStructSchema(std::string_view type,
                                   std::string_view schema) {
  writer_->AddSchema(type, "structschema", schema);
}

void WPILogWriter::Write(const control_loop::ContextInternal& context) {
  std::lock_guard lock(context.messages_mutex_);
  for (const auto& [channel, message] : context.messages_) {
    if (message == nullptr) {
      continue;
    }
    const auto handler = write_functions_.find(channel);
    if (handler != write_functions_.end()) {
      handler->second(*this, channel, *message);
    }
  }
}

void WPILogWriter::Close() {
  if (!closed_ && writer_ != nullptr) {
    writer_->Flush();
    writer_->Stop();
    closed_ = true;
  }
}
}  // namespace logging
