#pragma once
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
}  // namespace control_loop
