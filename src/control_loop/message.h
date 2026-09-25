#pragma once
#include <cstddef>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <typeindex>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace wpi::log { class DataLogWriter; }
namespace logging {
class FieldRegistrar;
using index_t = std::size_t;
template <typename T>
auto RegisterFields(wpi::log::DataLogWriter&, FieldRegistrar&)
    -> std::vector<index_t>;
}

namespace control_loop {

class IMessage {
 public:
  virtual ~IMessage() = default;
  IMessage() = default;

  // Interface
  virtual auto GetType() -> const std::type_info& = 0;
  virtual auto GetSize() -> std::size_t = 0;
};

template <typename T>
class ValueMessage final : public IMessage {
 public:
  explicit ValueMessage(T value) : value(std::move(value)) {}
  auto GetType() -> const std::type_info& override {
    return typeid(ValueMessage<T>);
  }
  auto GetSize() -> std::size_t override { return sizeof(*this); }

  T value;
};

class MessageDescriptor {
 public:
  using RegistrationFunction = std::function<std::vector<logging::index_t>(
      wpi::log::DataLogWriter&, logging::FieldRegistrar&)>;
  struct PublicationInfo {
    std::type_index message_type;
    std::type_index value_type;
    bool is_class;
    std::optional<RegistrationFunction> register_logs;
  };

  MessageDescriptor(std::string_view channel, std::type_index type)
      : channel_(channel), types_({type}) {}
  MessageDescriptor(std::string_view channel,
                    std::unordered_set<std::type_index> types)
      : channel_(channel), types_(std::move(types)) {}

  template <typename T>
  static auto Publication(std::string_view channel)
      -> MessageDescriptor {
    using Stored = std::conditional_t<std::is_base_of_v<IMessage, T>, T,
                                      ValueMessage<T>>;
    MessageDescriptor descriptor(channel, typeid(Stored));
    // LOG_FIELDS generates this method inside T. Its body delegates to the
    // single RegisterFields<T> implementation in log_registration.h.
    RegistrationFunction registration;
    if constexpr (requires(wpi::log::DataLogWriter& log,
                           logging::FieldRegistrar& registrar) {
                    T::RegisterWPILog(log, registrar);
                  }) {
      registration = &T::RegisterWPILog;
    }
    descriptor.publication_.emplace(PublicationInfo{
        typeid(Stored), typeid(T), std::is_class_v<T>,
        registration ? std::optional<RegistrationFunction>{
                           std::move(registration)}
                     : std::nullopt});
    return descriptor;
  }

  [[nodiscard]] auto GetChannel() const -> const std::string& {
    return channel_;
  }
  [[nodiscard]] auto GetTypes() const
      -> const std::unordered_set<std::type_index>& {
    return types_;
  }
  [[nodiscard]] auto GetPublicationInfo() const
      -> const std::optional<PublicationInfo>& {
    return publication_;
  }

 private:
  std::string channel_;
  std::unordered_set<std::type_index> types_;
  std::optional<PublicationInfo> publication_;
};

}  // namespace control_loop
