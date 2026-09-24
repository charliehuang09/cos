#pragma once

#include <chrono>
#include <cstdint>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <string_view>
#include <typeindex>
#include <unordered_map>
#include <vector>

#include <wpi/DataLogWriter.h>

#include "control_loop/context.h"

namespace logging {

// Contexts retain the writer until their last message is available.
class WPILogWriter final {
 public:
  explicit WPILogWriter(std::string path);
  ~WPILogWriter();

  void Configure(
      const std::vector<control_loop::MessageDescriptor>& publications);
  void Write(const control_loop::ContextInternal& context);

  int StartRaw(std::string_view channel, std::string_view type);
  int StartDouble(std::string_view channel);
  int StartIntegerArray(std::string_view channel);
  int StartDoubleArray(std::string_view channel);
  int StartString(std::string_view channel);
  void AddStructSchema(std::string_view type, std::string_view schema);
  void AddPose3dSchema();
  void AppendRaw(int entry, std::span<const uint8_t> value, int64_t timestamp);
  void AppendDouble(int entry, double value, int64_t timestamp);
  void AppendIntegerArray(int entry, std::span<const int64_t> values,
                          int64_t timestamp);
  void AppendDoubleArray(int entry, std::span<const double> values,
                         int64_t timestamp);
  void AppendString(int entry, std::string_view value, int64_t timestamp);
  // Call after the loop and its workers have released all contexts.
  void Close();

 private:
  struct LogEntry {
    std::vector<int> indices;
    std::type_index type;
    control_loop::MessageDescriptor::AppendWPILog append;
  };

  std::string path_;
  std::unique_ptr<wpi::log::DataLogWriter> writer_;
  std::unordered_map<std::string, LogEntry> entries_;
  std::mutex mutex_;
  std::chrono::steady_clock::time_point last_flush_;
  bool closed_ = false;
};

}  // namespace logging
