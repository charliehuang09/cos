#include "logging/wpilog_writer.h"

#include <chrono>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_set>
#include <utility>

#include <wpi/raw_ostream.h>

namespace logging {

WPILogWriter::WPILogWriter(
    std::string_view filename,
    const std::vector<control_loop::MessageDescriptor>& publications) {
  std::error_code error;
  log_ = std::make_unique<wpi::log::DataLogWriter>(filename, error);
  if (error) {
    throw std::system_error(error, "Cannot open WPILog file");
  }

  std::unordered_set<std::string> channels;
  std::unordered_set<std::string> seen_paths;
  // The file has one DataLogWriter, but every publication gets its own field
  // registrations. The channel is a runtime value from MessageDescriptor.
  for (const auto& publication : publications) {
    const auto& info = publication.GetPublicationInfo();
    if (!info.has_value() || publication.GetTypes().size() != 1 ||
        !publication.GetTypes().contains(info->message_type)) {
      throw std::invalid_argument("Publication lacks concrete type: " +
                                  publication.GetChannel());
    }
    if (!channels.insert(publication.GetChannel()).second) {
      throw std::invalid_argument("Duplicate publication: " +
                                  publication.GetChannel());
    }

    if (info->register_logs == nullptr) {
      throw std::invalid_argument("Missing WPILog registration: " +
                                  publication.GetChannel());
    }
    PublicationLog group{publication.GetChannel(), {}};
    info->register_logs(*log_, group.channel, group.fields);
    for (const auto& slot : group.fields) {
      if (!seen_paths.insert(slot.path).second) {
        throw std::invalid_argument("Duplicate WPILog path: " + slot.path);
      }
    }
    publications_.push_back(std::move(group));
  }

  flush_thread_ = std::jthread([this](std::stop_token stop_token) {
    std::unique_lock wait_lock(flush_wait_mutex_);
    while (!flush_cv_.wait_for(wait_lock, std::chrono::seconds(1),
                               [&] { return stop_token.stop_requested(); })) {
      Flush();
    }
  });
}

WPILogWriter::~WPILogWriter() {
  flush_thread_.request_stop();
  flush_cv_.notify_one();
  flush_thread_.join();
  publications_.clear();
  log_->Flush();
  log_->Stop();
}

auto WPILogWriter::GetLogPaths() const -> std::vector<std::string> {
  std::vector<std::string> paths;
  for (const auto& group : publications_) {
    for (const auto& slot : group.fields) paths.push_back(slot.path);
  }
  return paths;
}

void WPILogWriter::Log(const control_loop::ContextInternal& context) {
  std::lock_guard lock(mutex_);
  for (const auto& group : publications_) {
    const auto* message =
        context.GetMessage<control_loop::IMessage>(group.channel);
    if (message == nullptr) continue;
    for (const auto& slot : group.fields) {
      if (!slot.field->Append(*message)) {
        throw std::runtime_error("WPILog message type mismatch: " + slot.path);
      }
    }
  }
}

void WPILogWriter::Flush() {
  std::lock_guard lock(mutex_);
  log_->Flush();
  log_->GetStream().flush();
}

}  // namespace logging
