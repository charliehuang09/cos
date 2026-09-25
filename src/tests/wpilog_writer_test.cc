#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>
#include <wpi/DataLogReader.h>
#include <wpi/MemoryBuffer.h>

#include "control_loop/context.h"
#include "localization/position.h"
#include "logging/wpilog_writer.h"

namespace {

auto LogPath() -> std::filesystem::path {
  return std::filesystem::temp_directory_path() /
         ("cos-wpilog-writer-" + std::to_string(getpid()) + ".wpilog");
}

using Pose2dMessage = control_loop::ValueMessage<frc::Pose2d>;

auto RegisterPose2d(wpi::log::DataLogWriter& log,
                    logging::FieldRegistrar& registrar, std::string_view)
    -> std::vector<logging::index_t> {
  return {registrar.Add<Pose2dMessage>(
      log, "", [](const Pose2dMessage& message) { return message.value; })};
}

TEST(WPILogWriterTest, WritesRegisteredFieldsAndSkipsMissingMessages) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<double>("temperature"),
      control_loop::MessageDescriptor::Publication<
          localization::PositionEstimateMessage>(
          "pose", &localization::PositionEstimateMessage::RegisterWPILog),
  };

  {
    logging::WPILogWriter writer(path.string(), publications);
    ASSERT_EQ(writer.GetLogPaths().size(), 5);

    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                          nullptr, std::stop_token{}, 1);
    context.SetMessage(
        "temperature",
        std::make_unique<control_loop::ValueMessage<double>>(42.5));
    auto estimate =
        std::make_unique<localization::PositionEstimateMessage>();
    estimate->variance = 0.25;
    estimate->tag_ids = {3, 7};
    estimate->distances = {1.5, 2.5};
    context.SetMessage("pose", std::move(estimate));
    writer.Log(context);

    control_loop::ContextInternal empty(std::chrono::steady_clock::now(),
                                        nullptr, std::stop_token{}, 2);
    empty.SetMessage("pose", nullptr);
    writer.Log(empty);
    writer.Flush();
  }

  auto buffer = wpi::MemoryBuffer::GetFile(path.string());
  ASSERT_TRUE(buffer.has_value());
  wpi::log::DataLogReader reader(std::move(*buffer));
  ASSERT_TRUE(reader.IsValid());

  std::unordered_map<int, std::string> entry_names;
  std::unordered_map<std::string, int> value_counts;
  const std::unordered_set<std::string> expected_names = {
      "temperature", "pose/pose", "pose/variance", "pose/tag_ids",
      "pose/distances"};
  for (const auto& record : reader) {
    if (record.IsStart()) {
      wpi::log::StartRecordData start;
      ASSERT_TRUE(record.GetStartData(&start));
      entry_names.emplace(start.entry, start.name);
      continue;
    }
    if (record.IsControl()) {
      continue;
    }
    const auto& name = entry_names.at(record.GetEntry());
    if (!expected_names.contains(name)) {
      continue;
    }
    ++value_counts[name];
    if (name == "temperature") {
      double value = 0;
      ASSERT_TRUE(record.GetDouble(&value));
      EXPECT_DOUBLE_EQ(value, 42.5);
    } else if (name == "pose/variance") {
      double value = 0;
      ASSERT_TRUE(record.GetDouble(&value));
      EXPECT_DOUBLE_EQ(value, 0.25);
    } else if (name == "pose/tag_ids") {
      std::vector<std::int64_t> ids;
      ASSERT_TRUE(record.GetIntegerArray(&ids));
      EXPECT_EQ(ids, (std::vector<std::int64_t>{3, 7}));
    } else if (name == "pose/distances") {
      std::vector<double> distances;
      ASSERT_TRUE(record.GetDoubleArray(&distances));
      EXPECT_EQ(distances, (std::vector<double>{1.5, 2.5}));
    }
  }
  EXPECT_EQ(value_counts.size(), 5);
  for (const auto& [name, count] : value_counts) {
    EXPECT_EQ(count, 1) << name;
  }
  std::filesystem::remove(path);
}

TEST(WPILogWriterTest, RequiresRegistrationForClassPublications) {
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<
          localization::PositionEstimateMessage>("pose"),
  };
  EXPECT_THROW(logging::WPILogWriter(LogPath().string(), publications),
               std::invalid_argument);
  std::filesystem::remove(LogPath());
}

TEST(WPILogWriterTest, WritesBuiltInTypesAndPose2d) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<bool>("ready"),
      control_loop::MessageDescriptor::Publication<std::int64_t>("count"),
      control_loop::MessageDescriptor::Publication<std::string>("state"),
      control_loop::MessageDescriptor::Publication<frc::Pose2d>(
          "location", &RegisterPose2d),
  };
  {
    logging::WPILogWriter writer(path.string(), publications);
    ASSERT_EQ(writer.GetLogPaths().size(), 4);
    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                          nullptr, std::stop_token{}, 3);
    context.SetMessage(
        "ready", std::make_unique<control_loop::ValueMessage<bool>>(true));
    context.SetMessage(
        "count",
        std::make_unique<control_loop::ValueMessage<std::int64_t>>(17));
    context.SetMessage("state",
                       std::make_unique<control_loop::ValueMessage<std::string>>(
                           "tracking"));
    context.SetMessage(
        "location",
        std::make_unique<Pose2dMessage>(frc::Pose2d{}));
    writer.Log(context);
  }

  auto buffer = wpi::MemoryBuffer::GetFile(path.string());
  ASSERT_TRUE(buffer.has_value());
  wpi::log::DataLogReader reader(std::move(*buffer));
  ASSERT_TRUE(reader.IsValid());
  std::unordered_map<int, std::string> names;
  std::unordered_set<std::string> seen;
  for (const auto& record : reader) {
    if (record.IsStart()) {
      wpi::log::StartRecordData start;
      ASSERT_TRUE(record.GetStartData(&start));
      names.emplace(start.entry, start.name);
    } else if (!record.IsControl()) {
      const auto& name = names.at(record.GetEntry());
      if (name == "ready") {
        bool value = false;
        ASSERT_TRUE(record.GetBoolean(&value));
        EXPECT_TRUE(value);
      } else if (name == "count") {
        std::int64_t value = 0;
        ASSERT_TRUE(record.GetInteger(&value));
        EXPECT_EQ(value, 17);
      } else if (name == "state") {
        std::string_view value;
        ASSERT_TRUE(record.GetString(&value));
        EXPECT_EQ(value, "tracking");
      } else if (name == "location") {
        EXPECT_FALSE(record.GetRaw().empty());
      } else {
        continue;
      }
      EXPECT_TRUE(seen.insert(name).second);
    }
  }
  EXPECT_EQ(seen.size(), 4);
  std::filesystem::remove(path);
}

}  // namespace
