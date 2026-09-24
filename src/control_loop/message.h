#pragma once
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>
#include <typeindex>
#include <unordered_set>
#include <utility>
#include <vector>

namespace logging {
class WPILogWriter;
}

namespace control_loop {

class IMessage {
 public:
  static constexpr bool loggable = true;
  virtual ~IMessage() = default;
  IMessage() = default;

  // Interface
  virtual auto GetType() -> const std::type_info& = 0;
  virtual auto GetSize() -> std::size_t = 0;
};

class MessageDescriptor {
 public:
  using StartWPILog = std::vector<int> (*)(logging::WPILogWriter&,
                                          std::string_view);
  using AppendWPILog = void (*)(logging::WPILogWriter&, std::span<const int>,
                               const IMessage&, int64_t);

  template <typename T>
  static auto For(std::string_view channel) -> MessageDescriptor;

  MessageDescriptor(std::string_view channel, std::type_index type)
      : channel_(channel), types_({type}) {}
  MessageDescriptor(std::string_view channel, std::type_index type,
                    StartWPILog start_wpilog, AppendWPILog append_wpilog)
      : channel_(channel),
        types_({type}),
        start_wpilog_(start_wpilog),
        append_wpilog_(append_wpilog) {}
  MessageDescriptor(std::string_view channel,
                    std::unordered_set<std::type_index> types)
      : channel_(channel), types_(std::move(types)) {}
  [[nodiscard]] auto GetChannel() const -> const std::string& {
    return channel_;
  }
  [[nodiscard]] auto GetTypes() const
      -> const std::unordered_set<std::type_index>& {
    return types_;
  }
  [[nodiscard]] auto GetStartWPILog() const -> StartWPILog {
    return start_wpilog_;
  }
  [[nodiscard]] auto GetAppendWPILog() const -> AppendWPILog {
    return append_wpilog_;
  }

 private:
  std::string channel_;
  std::unordered_set<std::type_index> types_;
  StartWPILog start_wpilog_ = nullptr;
  AppendWPILog append_wpilog_ = nullptr;
};

}  // namespace control_loop
