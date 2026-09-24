#pragma once

#include "logging/publication.h"

namespace localization {
struct PositionEstimateMessage;
struct VarianceMessage;
class AmbiguousEstimateMessage;
}

namespace logging {

template <>
std::vector<int> StartLog<localization::PositionEstimateMessage>(
    WPILogWriter& writer, std::string_view channel);

template <>
void AppendLog<localization::PositionEstimateMessage>(
    WPILogWriter& writer, std::span<const int> entries,
    const localization::PositionEstimateMessage& message, int64_t timestamp);

template <>
std::vector<int> StartLog<localization::VarianceMessage>(
    WPILogWriter& writer, std::string_view channel);

template <>
void AppendLog<localization::VarianceMessage>(
    WPILogWriter& writer, std::span<const int> entries,
    const localization::VarianceMessage& message, int64_t timestamp);

template <>
std::vector<int> StartLog<localization::AmbiguousEstimateMessage>(
    WPILogWriter& writer, std::string_view channel);

template <>
void AppendLog<localization::AmbiguousEstimateMessage>(
    WPILogWriter& writer, std::span<const int> entries,
    const localization::AmbiguousEstimateMessage& message, int64_t timestamp);

}  // namespace logging
