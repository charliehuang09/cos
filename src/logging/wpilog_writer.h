#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "control_loop/context.h"
#include "logging/log_registration.h"

namespace logging {

class WPILogWriter {
 public:
  WPILogWriter(std::string_view filename,
               const std::vector<control_loop::MessageDescriptor>& publications);
  ~WPILogWriter();

  WPILogWriter(const WPILogWriter&) = delete;
  auto operator=(const WPILogWriter&) -> WPILogWriter& = delete;

  void Log(const control_loop::ContextInternal& context);
  void Flush();

  [[nodiscard]] auto GetLogPaths() const -> std::vector<std::string>;

 private:
  struct PublicationLog {
    std::string channel;
    std::vector<FieldSlot> fields;
  };

  std::unique_ptr<wpi::log::DataLogWriter> log_;
  std::vector<PublicationLog> publications_;
  std::mutex mutex_;
  std::mutex flush_wait_mutex_;
  std::condition_variable flush_cv_;
  std::jthread flush_thread_;
};

}  // namespace logging
