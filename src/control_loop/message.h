#pragma once
#include <string>
#include <string_view>
#include <typeinfo>

namespace control_loop {

class IMessage {
 public:
  virtual ~IMessage() = default;

  IMessage() = default;

  IMessage(const IMessage&) = delete;
  auto operator=(const IMessage&) -> IMessage& = delete;

  // Interface
  virtual auto GetType() -> const std::type_info& = 0;
};

class FailedMessage final : public IMessage {
 public:
  FailedMessage(std::string_view source, std::string_view reason)
      : source(source), reason(reason) {}

  auto GetType() -> const std::type_info& override {
    return typeid(FailedMessage);
  }

  std::string source;
  std::string reason;
};
}  // namespace control_loop
