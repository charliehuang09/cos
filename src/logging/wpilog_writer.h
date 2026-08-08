#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include <wpi/DataLogWriter.h>

namespace logging {

// A terminal node that records the messages accumulated in a context.
// Messages over max_message_bytes are omitted.
class WPILogWriter final {
 public:
  WPILogWriter();

  template <typename T>
  void RegisterLoggable(const std::string& channel);

  template<typename T>
  void Write(const std::string& channel);

  // Flushes all buffered records and closes the WPILOG file. No records are
  // accepted after this returns.
  void Close();

 private:
  std::unique_ptr<wpi::log::DataLogWriter> writer_;
  std::unordered_map<std::string, int> entry_indices_;
};

}  // namespace logging
