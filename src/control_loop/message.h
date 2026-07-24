#pragma once
#include <cstddef>
#include <typeinfo>
namespace control_loop {

class IMessage {
 public:
  virtual ~IMessage() = default;
  IMessage() = default;

  // Interface
  virtual auto GetType() -> const std::type_info& = 0;
  virtual auto GetSize() -> std::size_t = 0;
};
}  // namespace control_loop
