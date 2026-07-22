#pragma once
#include <cstddef>
#include <string>
#include <typeindex>
#include <unordered_set>
#include <utility>

namespace control_loop {

class IMessage {
 public:
  virtual ~IMessage() = default;
  IMessage() = default;

  // Interface
  virtual auto GetType() -> const std::type_info& = 0;
  virtual auto GetSize() -> std::size_t = 0;
};

class MessageDescriptor {
 public:
  MessageDescriptor(std::string_view channel, std::type_index type)
      : channel_(channel), types_({type}) {}
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

 private:
  std::string channel_;
  std::unordered_set<std::type_index> types_;
};

}  // namespace control_loop
