#include "logging/wpilog_writer.h"

#include <system_error>
#include <utility>

#include <wpi/timestamp.h>
#include <frc/geometry/struct/Pose3dStruct.h>

#include "absl/log/check.h"
#include "absl/log/log.h"

namespace logging {
namespace {
constexpr auto kFlushInterval = std::chrono::milliseconds(250);
}  // namespace

WPILogWriter::WPILogWriter(std::string path) : path_(std::move(path)) {}

WPILogWriter::~WPILogWriter() {
  Close();
}

void WPILogWriter::Close() {
  std::lock_guard lock(mutex_);
  if (writer_ != nullptr && !closed_) {
    writer_->Flush();
    writer_->Stop();
    closed_ = true;
  }
}

void WPILogWriter::Configure(
    const std::vector<control_loop::MessageDescriptor>& publications) {
  std::lock_guard lock(mutex_);
  CHECK(writer_ == nullptr) << "WPILogWriter was configured more than once";
  std::error_code ec;
  writer_ = std::make_unique<wpi::log::DataLogWriter>(path_, ec);
  CHECK(!ec) << "Failed to create WPILOG at " << path_ << ": " << ec.message();

  for (const auto& publication : publications) {
    const auto start = publication.GetStartWPILog();
    if (start == nullptr) {
      continue;
    }
    CHECK_EQ(publication.GetTypes().size(), 1U);
    const auto type = *publication.GetTypes().begin();
    auto entries = start(*this, publication.GetChannel());
    entries_.emplace(publication.GetChannel(),
                     LogEntry{std::move(entries), type,
                              publication.GetAppendWPILog()});
  }
  last_flush_ = std::chrono::steady_clock::now();
}

void WPILogWriter::Write(
    const control_loop::ContextInternal& context) {
  std::lock_guard lock(mutex_);
  if (writer_ == nullptr || closed_) {
    return;
  }
  const int64_t timestamp = wpi::Now();
  std::lock_guard context_lock(context.messages_mutex_);
  for (const auto& [channel, message] : context.messages_) {
    if (message == nullptr) {
      continue;
    }
    const auto entry = entries_.find(channel);
    if (entry == entries_.end()) {
      continue;
    }
    if (std::type_index(typeid(*message)) != entry->second.type) {
      LOG(ERROR) << "WPILOG message type differs from publication on "
                 << channel;
      continue;
    }
    entry->second.append(*this, entry->second.indices, *message, timestamp);
  }
  const auto now = std::chrono::steady_clock::now();
  if (now - last_flush_ >= kFlushInterval) {
    writer_->Flush();
    last_flush_ = now;
  }
}

int WPILogWriter::StartRaw(std::string_view channel, std::string_view type) {
  return writer_->Start(channel, type);
}

int WPILogWriter::StartDouble(std::string_view channel) {
  return writer_->Start(channel, "double");
}

int WPILogWriter::StartIntegerArray(std::string_view channel) {
  return writer_->Start(channel, "int64[]");
}

int WPILogWriter::StartDoubleArray(std::string_view channel) {
  return writer_->Start(channel, "double[]");
}

int WPILogWriter::StartString(std::string_view channel) {
  return writer_->Start(channel, "string");
}

void WPILogWriter::AddStructSchema(std::string_view type,
                                   std::string_view schema) {
  writer_->AddSchema(type, "structschema", schema);
}

void WPILogWriter::AddPose3dSchema() {
  writer_->AddStructSchema<frc::Pose3d>();
}

void WPILogWriter::AppendRaw(int entry, std::span<const uint8_t> value,
                             int64_t timestamp) {
  writer_->AppendRaw(entry, value, timestamp);
}

void WPILogWriter::AppendDouble(int entry, double value, int64_t timestamp) {
  writer_->AppendDouble(entry, value, timestamp);
}

void WPILogWriter::AppendIntegerArray(int entry,
                                     std::span<const int64_t> values,
                                     int64_t timestamp) {
  writer_->AppendIntegerArray(entry, values, timestamp);
}

void WPILogWriter::AppendDoubleArray(int entry,
                                    std::span<const double> values,
                                    int64_t timestamp) {
  writer_->AppendDoubleArray(entry, values, timestamp);
}

void WPILogWriter::AppendString(int entry, std::string_view value,
                                int64_t timestamp) {
  writer_->AppendString(entry, value, timestamp);
}

}  // namespace logging
