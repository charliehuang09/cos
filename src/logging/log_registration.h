#pragma once

#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <limits>
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
#include <wpi/DataLog.h>
#include <wpi/DataLogWriter.h>

#include "control_loop/message.h"

namespace logging {

namespace detail {
template <typename T>
struct LogEntryType;

#define COS_LOG_ENTRY(Value, Entry) \
  template <> struct LogEntryType<Value> { using type = wpi::log::Entry; };
COS_LOG_ENTRY(bool, BooleanLogEntry)
COS_LOG_ENTRY(std::int64_t, IntegerLogEntry)
COS_LOG_ENTRY(float, FloatLogEntry)
COS_LOG_ENTRY(double, DoubleLogEntry)
COS_LOG_ENTRY(std::string, StringLogEntry)
COS_LOG_ENTRY(std::vector<std::string>, StringArrayLogEntry)
COS_LOG_ENTRY(std::vector<std::int64_t>, IntegerArrayLogEntry)
COS_LOG_ENTRY(std::vector<double>, DoubleArrayLogEntry)
COS_LOG_ENTRY(std::vector<int>, BooleanArrayLogEntry)
COS_LOG_ENTRY(frc::Pose2d, StructLogEntry<frc::Pose2d>)
COS_LOG_ENTRY(frc::Pose3d, StructLogEntry<frc::Pose3d>)
COS_LOG_ENTRY(std::vector<frc::Pose2d>, StructArrayLogEntry<frc::Pose2d>)
COS_LOG_ENTRY(std::vector<frc::Pose3d>, StructArrayLogEntry<frc::Pose3d>)
#undef COS_LOG_ENTRY

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
inline constexpr bool IsNativeArrayElement =
    std::is_arithmetic_v<T> || std::is_enum_v<T> ||
    std::is_same_v<T, std::string> || std::is_same_v<T, frc::Pose2d> ||
    std::is_same_v<T, frc::Pose3d>;
template <typename T>
struct IsIgnoredField : std::false_type {};
template <typename T, typename Allocator>
struct IsIgnoredField<std::vector<T, Allocator>>
    : std::bool_constant<!IsNativeArrayElement<T>> {};
template <typename T, std::size_t N>
struct IsIgnoredField<std::array<T, N>>
    : std::bool_constant<!IsNativeArrayElement<T>> {};
template <typename T>
struct IsOptional : std::false_type {};
template <typename T>
struct IsOptional<std::optional<T>> : std::true_type {};

template <typename T>
auto NormalizeLogValue(const T& value) {
  if constexpr ((std::is_integral_v<T> && !std::is_same_v<T, bool>) ||
                std::is_enum_v<T>) {
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
  } else if constexpr (IsArray<T>::value || IsVector<T>::value) {
    using Element = typename T::value_type;
    using Normalized = std::conditional_t<std::is_same_v<Element, bool>, int,
        std::conditional_t<std::is_floating_point_v<Element>, double,
            decltype(NormalizeLogValue(std::declval<const Element&>()))>>;
    std::vector<Normalized> result;
    result.reserve(value.size());
    for (const auto& item : value) {
      result.push_back(static_cast<Normalized>(NormalizeLogValue(item)));
    }
    return result;
  } else {
    static_assert(requires { typename LogEntryType<T>::type; },
                  "Unsupported WPILog field type");
    return value;
  }
}

template <typename Root, typename Getter>
void RegisterValue(wpi::log::DataLogWriter& log, const std::string& path,
                   Getter getter, std::vector<std::string>& paths,
                   std::vector<std::move_only_function<void(const Root&)>>& fields) {
  using Value = std::remove_cvref_t<std::invoke_result_t<Getter, const Root&>>;
  if constexpr (IsOptional<Value>::value) {
    using Element = typename Value::value_type;
    RegisterValue<Root>(log, path + "_present",
        [getter](const Root& message) { return getter(message).has_value(); },
        paths, fields);
    RegisterValue<Root>(log, path,
        [getter](const Root& message) -> const Element& {
          const auto& value = getter(message);
          static const Element empty{};
          return value ? *value : empty;
        }, paths, fields);
  } else if constexpr (requires { Value::WpiLogFields(); }) {
    std::apply([&](auto... members) {
      (RegisterValue<Root>(log, path + '/' + members.first,
          [getter, pointer = members.second](const Root& message) -> const auto& {
            return getter(message).*pointer;
          }, paths, fields), ...);
    }, Value::WpiLogFields());
  } else if constexpr (IsArray<Value>::value &&
                       requires { Value::value_type::WpiLogFields(); }) {
    for (std::size_t i = 0; i < std::tuple_size_v<Value>; ++i) {
      RegisterValue<Root>(log, path + '/' + std::to_string(i),
          [getter, i](const Root& message) -> const auto& {
            return getter(message)[i];
          }, paths, fields);
    }
  } else if constexpr (!IsIgnoredField<Value>::value) {
    using Normalized = decltype(NormalizeLogValue(std::declval<const Value&>()));
    using Entry = typename LogEntryType<Normalized>::type;
    paths.push_back(path);
    fields.emplace_back([entry = Entry(log, path), getter = std::move(getter)](
                            const Root& message) mutable {
      entry.Append(NormalizeLogValue(getter(message)));
    });
  }
}
}  // namespace detail

template <typename T>
auto RegisterFields(wpi::log::DataLogWriter& log, std::string_view channel,
                    std::vector<std::string>& paths) -> LogFunction {
  using Message = std::conditional_t<std::is_base_of_v<control_loop::IMessage, T>,
                                     T, control_loop::ValueMessage<T>>;
  auto root = [](const Message& message) -> const T& {
    if constexpr (std::is_same_v<T, Message>) return message;
    else return message.value;
  };
  std::vector<std::move_only_function<void(const Message&)>> appenders;
  detail::RegisterValue<Message>(log, std::string(channel), root, paths, appenders);
  return [appenders = std::move(appenders)](
             const control_loop::IMessage& message) mutable {
    const auto* typed = dynamic_cast<const Message*>(&message);
    if (typed == nullptr) return false;
    for (auto& append : appenders) append(*typed);
    return true;
  };
}

}  // namespace logging

#define COS_LOG_MEMBER(Type, member) std::pair{#member, &Type::member}
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

// Annotated messages register a callback for each runtime publication channel.
#define LOG_FIELDS(Type, ...)                                     \
  static constexpr auto WpiLogFields() {                          \
    return std::make_tuple(COS_LOG_MEMBERS(Type, __VA_ARGS__));     \
  }                                                               \
  static constexpr auto RegisterWPILog = &::logging::RegisterFields<Type>;
