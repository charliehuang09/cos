#pragma once

#include <cstdint>
#include <array>
#include <chrono>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include <frc/geometry/Pose2d.h>
#include <frc/geometry/Pose3d.h>
#include <frc/geometry/struct/Pose2dStruct.h>
#include <frc/geometry/struct/Pose3dStruct.h>
#include <nlohmann/json.hpp>
#include <wpi/DataLog.h>
#include <wpi/DataLogWriter.h>

#include "control_loop/message.h"

namespace logging {

class ILogField {
 public:
  virtual ~ILogField() = default;
  virtual auto Append(const control_loop::IMessage& message) -> bool = 0;
};

namespace detail {
template <typename T>
struct LogEntryType;

template <>
struct LogEntryType<bool> {
  using type = wpi::log::BooleanLogEntry;
};
template <>
struct LogEntryType<std::int64_t> {
  using type = wpi::log::IntegerLogEntry;
};
template <>
struct LogEntryType<float> {
  using type = wpi::log::FloatLogEntry;
};
template <>
struct LogEntryType<double> {
  using type = wpi::log::DoubleLogEntry;
};
template <>
struct LogEntryType<std::string> {
  using type = wpi::log::StringLogEntry;
};
template <>
struct LogEntryType<std::vector<std::string>> {
  using type = wpi::log::StringArrayLogEntry;
};
template <>
struct LogEntryType<std::vector<std::int64_t>> {
  using type = wpi::log::IntegerArrayLogEntry;
};
template <>
struct LogEntryType<std::vector<double>> {
  using type = wpi::log::DoubleArrayLogEntry;
};
template <>
struct LogEntryType<std::vector<int>> {
  using type = wpi::log::BooleanArrayLogEntry;
};
template <>
struct LogEntryType<frc::Pose2d> {
  using type = wpi::log::StructLogEntry<frc::Pose2d>;
};
template <>
struct LogEntryType<frc::Pose3d> {
  using type = wpi::log::StructLogEntry<frc::Pose3d>;
};
template <>
struct LogEntryType<std::vector<frc::Pose2d>> {
  using type = wpi::log::StructArrayLogEntry<frc::Pose2d>;
};
template <>
struct LogEntryType<std::vector<frc::Pose3d>> {
  using type = wpi::log::StructArrayLogEntry<frc::Pose3d>;
};

template <typename Message, typename Value>
class TypedLogField final : public ILogField {
 public:
  template <typename Getter>
  TypedLogField(wpi::log::DataLogWriter& log, std::string_view path,
                Getter getter)
      // The WPILib entry is constructed once, at writer startup. Append()
      // later uses this same entry object; it does not look up a path again.
      : entry_(log, path), getter_(std::move(getter)) {}

  auto Append(const control_loop::IMessage& message) -> bool override {
    const auto* typed = dynamic_cast<const Message*>(&message);
    if (typed == nullptr) {
      return false;
    }
    entry_.Append(getter_(*typed));
    return true;
  }

 private:
  typename LogEntryType<Value>::type entry_;
  std::function<Value(const Message&)> getter_;
};
}  // namespace detail

class FieldRegistrar {
 public:
  using AddFieldFunction = std::function<index_t(
      std::string, std::string, std::unique_ptr<ILogField>)>;

  FieldRegistrar(std::string_view channel, AddFieldFunction add_field)
      : channel_(channel), add_field_(std::move(add_field)) {}

  template <typename Message, typename Getter>
  auto Add(wpi::log::DataLogWriter& log, std::string_view suffix, Getter getter)
      -> index_t {
    using Value = std::decay_t<std::invoke_result_t<Getter, const Message&>>;
    // channel_ comes from this particular MessageDescriptor. For example,
    // "master/sub1" plus "d/x" becomes the WPILib entry "master/sub1/d/x".
    std::string path = channel_;
    if (!suffix.empty()) {
      path += '/';
      path += suffix;
    }
    auto field = std::make_unique<detail::TypedLogField<Message, Value>>(
        log, path, std::move(getter));
    return add_field_(std::move(path), channel_, std::move(field));
  }

 private:
  std::string channel_;
  AddFieldFunction add_field_;
};

template <typename Owner, typename Value>
struct LogMember {
  // This is metadata only: the field name and a pointer to its member.
  std::string_view name;
  Value Owner::* member;
};
template <typename Owner, typename Value>
LogMember(std::string_view, Value Owner::*) -> LogMember<Owner, Value>;

namespace detail {
template <typename T>
struct IsVector : std::false_type {};
template <typename T, typename Allocator>
struct IsVector<std::vector<T, Allocator>> : std::true_type {};
template <typename T>
struct IsArray : std::false_type {};
template <typename T, std::size_t N>
struct IsArray<std::array<T, N>> : std::true_type {};
template <typename T>
struct IsDuration : std::false_type {};
template <typename Rep, typename Period>
struct IsDuration<std::chrono::duration<Rep, Period>> : std::true_type {};
template <typename T>
struct IsOptional : std::false_type {};
template <typename T>
struct IsOptional<std::optional<T>> : std::true_type {};
template <typename T>
concept HasLogFields = requires { T::WpiLogFields(); };
template <typename T>
concept HasNumericXY = requires(const T& value) {
  value.x;
  value.y;
  requires std::is_arithmetic_v<std::remove_cvref_t<decltype(value.x)>>;
  requires std::is_arithmetic_v<std::remove_cvref_t<decltype(value.y)>>;
};
template <typename>
inline constexpr bool always_false = false;

template <typename T>
auto ToJson(const T& value) -> nlohmann::json {
  // A vector of annotated structs is one variable-length JSON entry. Its
  // nested names still come solely from LOG_FIELDS, not per-struct code.
  if constexpr (std::is_same_v<T, frc::Pose3d>) {
    const auto& translation = value.Translation();
    const auto& rotation = value.Rotation();
    return {{"x", translation.X().value()},
            {"y", translation.Y().value()},
            {"z", translation.Z().value()},
            {"roll", rotation.X().value()},
            {"pitch", rotation.Y().value()},
            {"yaw", rotation.Z().value()}};
  } else if constexpr (std::is_same_v<T, frc::Pose2d>) {
    return {{"x", value.X().value()},
            {"y", value.Y().value()},
            {"angle", value.Rotation().Radians().value()}};
  } else if constexpr (HasLogFields<T>) {
    auto result = nlohmann::json::object();
    std::apply(
        [&](auto... fields) {
          ((result[std::string(fields.name)] =
                ToJson(value.*fields.member)), ...);
        },
        T::WpiLogFields());
    return result;
  } else if constexpr (IsOptional<T>::value) {
    return value ? ToJson(*value) : nlohmann::json(nullptr);
  } else if constexpr (IsVector<T>::value || IsArray<T>::value) {
    auto result = nlohmann::json::array();
    for (const auto& item : value) {
      if constexpr (std::is_same_v<typename T::value_type, bool>) {
        result.push_back(static_cast<bool>(item));
      } else {
        result.push_back(ToJson(item));
      }
    }
    return result;
  } else if constexpr (HasNumericXY<T>) {
    return {{"x", value.x}, {"y", value.y}};
  } else if constexpr (std::is_arithmetic_v<T> || std::is_enum_v<T> ||
                       std::is_same_v<T, std::string>) {
    return value;
  } else {
    static_assert(always_false<T>, "Unsupported nested WPILog JSON field");
  }
}

template <typename T>
auto NormalizeLogValue(const T& value) {
  // Every leaf becomes a type accepted by one of the LogEntryType mappings
  // above. In particular, WPILib integer entries use int64_t.
  if constexpr (std::is_same_v<T, bool> || std::is_same_v<T, float> ||
                std::is_same_v<T, double> || std::is_same_v<T, std::string> ||
                std::is_same_v<T, frc::Pose2d> ||
                std::is_same_v<T, frc::Pose3d>) {
    return value;
  } else if constexpr (std::is_integral_v<T> || std::is_enum_v<T>) {
    if constexpr (std::is_unsigned_v<T> && sizeof(T) >= sizeof(std::int64_t)) {
      if (value > static_cast<T>(std::numeric_limits<std::int64_t>::max())) {
        throw std::out_of_range("WPILog integer exceeds int64_t");
      }
    }
    return static_cast<std::int64_t>(value);
  } else if constexpr (std::is_same_v<T, long double>) {
    return static_cast<double>(value);
  } else if constexpr (IsDuration<T>::value) {
    return std::chrono::duration<double>(value).count();
  } else if constexpr (IsOptional<T>::value) {
    return ToJson(value).dump();
  } else if constexpr (IsArray<T>::value) {
    return NormalizeLogValue(std::vector<typename T::value_type>(value.begin(),
                                                                  value.end()));
  } else if constexpr (IsVector<T>::value) {
    using Element = typename T::value_type;
    if constexpr (std::is_same_v<Element, bool>) {
      std::vector<int> result;
      result.reserve(value.size());
      for (bool item : value) result.push_back(item ? 1 : 0);
      return result;
    } else if constexpr (std::is_integral_v<Element> ||
                         std::is_enum_v<Element>) {
      std::vector<std::int64_t> result;
      result.reserve(value.size());
      for (auto item : value) result.push_back(NormalizeLogValue(item));
      return result;
    } else if constexpr (std::is_floating_point_v<Element>) {
      std::vector<double> result;
      result.reserve(value.size());
      for (auto item : value) result.push_back(static_cast<double>(item));
      return result;
    } else if constexpr (std::is_same_v<Element, std::string> ||
                         std::is_same_v<Element, frc::Pose2d> ||
                         std::is_same_v<Element, frc::Pose3d>) {
      return value;
    } else if constexpr (HasLogFields<Element> || HasNumericXY<Element>) {
      return ToJson(value).dump();
    } else {
      static_assert(always_false<T>, "Unsupported WPILog array element");
    }
  } else {
    static_assert(always_false<T>, "Unsupported WPILog field type");
  }
}

template <typename Root, typename Current, typename Accessor>
void RegisterMembers(wpi::log::DataLogWriter& log, FieldRegistrar& registrar,
                     std::string_view prefix, Accessor accessor,
                     std::vector<index_t>& indices);

template <typename Root, typename Current, typename Accessor, typename Field>
void RegisterMember(wpi::log::DataLogWriter& log, FieldRegistrar& registrar,
                    std::string_view prefix, Accessor accessor, Field field,
                    std::vector<index_t>& indices) {
  using Value = std::remove_cvref_t<decltype(std::declval<Current>().*field.member)>;
  std::string suffix(prefix);
  if (!suffix.empty()) suffix += '/';
  suffix += field.name;
  auto child = [accessor, member = field.member](const Root& message)
      -> const Value& { return accessor(message).*member; };
  if constexpr (HasLogFields<Value>) {
    // A nested annotated struct contributes paths such as "d/x".
    RegisterMembers<Root, Value>(log, registrar, suffix, child, indices);
  } else {
    // Add constructs and stores one WPILib entry for this leaf field.
    // child is retained as the getter used by TypedLogField::Append().
    indices.push_back(registrar.Add<Root>(
        log, suffix, [child](const Root& message) {
          return NormalizeLogValue(child(message));
        }));
  }
}

template <typename Root, typename Current, typename Accessor>
void RegisterMembers(wpi::log::DataLogWriter& log, FieldRegistrar& registrar,
                     std::string_view prefix, Accessor accessor,
                     std::vector<index_t>& indices) {
  std::apply(
      [&](auto... fields) {
        (RegisterMember<Root, Current>(log, registrar, prefix, accessor,
                                       fields, indices), ...);
      },
      Current::WpiLogFields());
}
}  // namespace detail

template <typename T>
auto RegisterFields(wpi::log::DataLogWriter& log, FieldRegistrar& registrar)
    -> std::vector<index_t> {
  // Plain structs are carried by ValueMessage<T>; IMessage subclasses are
  // used directly. The rest of the traversal is identical for both.
  using Message = std::conditional_t<std::is_base_of_v<control_loop::IMessage, T>,
                                     T, control_loop::ValueMessage<T>>;
  auto root = [](const Message& message) -> const T& {
    if constexpr (std::is_same_v<T, Message>) return message;
    else return message.value;
  };
  std::vector<index_t> indices;
  detail::RegisterMembers<Message, T>(log, registrar, "", root, indices);
  return indices;
}

}  // namespace logging

#define COS_LOG_MEMBER(Type, member) ::logging::LogMember{#member, &Type::member}
#define COS_LOG_MEMBERS_1(T, a) COS_LOG_MEMBER(T, a)
#define COS_LOG_MEMBERS_2(T, a, b) COS_LOG_MEMBERS_1(T, a), COS_LOG_MEMBER(T, b)
#define COS_LOG_MEMBERS_3(T, a, b, c) COS_LOG_MEMBERS_2(T, a, b), COS_LOG_MEMBER(T, c)
#define COS_LOG_MEMBERS_4(T, a, b, c, d) COS_LOG_MEMBERS_3(T, a, b, c), COS_LOG_MEMBER(T, d)
#define COS_LOG_MEMBERS_5(T, a, b, c, d, e) COS_LOG_MEMBERS_4(T, a, b, c, d), COS_LOG_MEMBER(T, e)
#define COS_LOG_MEMBERS_6(T, a, b, c, d, e, f) COS_LOG_MEMBERS_5(T, a, b, c, d, e), COS_LOG_MEMBER(T, f)
#define COS_LOG_MEMBERS_7(T, a, b, c, d, e, f, g) COS_LOG_MEMBERS_6(T, a, b, c, d, e, f), COS_LOG_MEMBER(T, g)
#define COS_LOG_MEMBERS_8(T, a, b, c, d, e, f, g, h) COS_LOG_MEMBERS_7(T, a, b, c, d, e, f, g), COS_LOG_MEMBER(T, h)
#define COS_LOG_MEMBERS_9(T, a, b, c, d, e, f, g, h, i) COS_LOG_MEMBERS_8(T, a, b, c, d, e, f, g, h), COS_LOG_MEMBER(T, i)
#define COS_LOG_MEMBERS_10(T, a, b, c, d, e, f, g, h, i, j) COS_LOG_MEMBERS_9(T, a, b, c, d, e, f, g, h, i), COS_LOG_MEMBER(T, j)
#define COS_LOG_MEMBERS_11(T, a, b, c, d, e, f, g, h, i, j, k) COS_LOG_MEMBERS_10(T, a, b, c, d, e, f, g, h, i, j), COS_LOG_MEMBER(T, k)
#define COS_LOG_MEMBERS_12(T, a, b, c, d, e, f, g, h, i, j, k, l) COS_LOG_MEMBERS_11(T, a, b, c, d, e, f, g, h, i, j, k), COS_LOG_MEMBER(T, l)
#define COS_LOG_MEMBERS_13(T, a, b, c, d, e, f, g, h, i, j, k, l, m) COS_LOG_MEMBERS_12(T, a, b, c, d, e, f, g, h, i, j, k, l), COS_LOG_MEMBER(T, m)
#define COS_LOG_MEMBERS_14(T, a, b, c, d, e, f, g, h, i, j, k, l, m, n) COS_LOG_MEMBERS_13(T, a, b, c, d, e, f, g, h, i, j, k, l, m), COS_LOG_MEMBER(T, n)
#define COS_LOG_MEMBERS_15(T, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o) COS_LOG_MEMBERS_14(T, a, b, c, d, e, f, g, h, i, j, k, l, m, n), COS_LOG_MEMBER(T, o)
#define COS_LOG_MEMBERS_16(T, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p) COS_LOG_MEMBERS_15(T, a, b, c, d, e, f, g, h, i, j, k, l, m, n, o), COS_LOG_MEMBER(T, p)
#define COS_LOG_SELECT(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, _13, _14, _15, _16, NAME, ...) NAME
#define COS_LOG_MEMBERS(T, ...) COS_LOG_SELECT(__VA_ARGS__, COS_LOG_MEMBERS_16, COS_LOG_MEMBERS_15, COS_LOG_MEMBERS_14, COS_LOG_MEMBERS_13, COS_LOG_MEMBERS_12, COS_LOG_MEMBERS_11, COS_LOG_MEMBERS_10, COS_LOG_MEMBERS_9, COS_LOG_MEMBERS_8, COS_LOG_MEMBERS_7, COS_LOG_MEMBERS_6, COS_LOG_MEMBERS_5, COS_LOG_MEMBERS_4, COS_LOG_MEMBERS_3, COS_LOG_MEMBERS_2, COS_LOG_MEMBERS_1)(T, __VA_ARGS__)

// Place inside Type, after the fields. For example, LOG_FIELDS(Sample, a, d)
// expands to two methods inside Sample:
//   WpiLogFields() -> tuple{name "a", &Sample::a; name "d", &Sample::d}
//   RegisterWPILog(log, registrar) -> RegisterFields<Sample>(...)
// The generated method contains no field-specific logging logic. The
// descriptor calls it once per publication channel at writer startup. The
// registrar carries that publication's runtime channel.
#define LOG_FIELDS(Type, ...)                                               \
  static constexpr auto WpiLogFields() {                                    \
    return std::tuple{COS_LOG_MEMBERS(Type, __VA_ARGS__)};                   \
  }                                                                         \
  static auto RegisterWPILog(wpi::log::DataLogWriter& log,                  \
                            ::logging::FieldRegistrar& registrar)          \
      -> std::vector<::logging::index_t> {                                   \
    return ::logging::RegisterFields<Type>(log, registrar);                 \
  }
