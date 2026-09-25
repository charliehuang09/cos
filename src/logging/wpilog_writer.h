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

  [[nodiscard]] auto GetLogPaths() const
      -> const std::vector<std::pair<std::string, index_t>>& {
    return log_paths_;
  }

 private:
  struct Slot {
    std::string channel;
    std::unique_ptr<ILogField> field;
  };

  auto AddField(std::string path, std::string channel,
                std::unique_ptr<ILogField> field) -> index_t;
  void RegisterPrimitive(const control_loop::MessageDescriptor& publication,
                         wpi::log::DataLogWriter& log);

  std::unique_ptr<wpi::log::DataLogWriter> log_;
  std::vector<Slot> slots_;
  std::vector<std::pair<std::string, index_t>> log_paths_;
  std::mutex mutex_;
  std::mutex flush_wait_mutex_;
  std::condition_variable flush_cv_;
  std::jthread flush_thread_;
};

}  // namespace logging
