#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <thread>
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

struct Pose2dSample {
  frc::Pose2d pose;
  LOG_FIELDS(Pose2dSample, pose)
};

struct UnregisteredSample {
  int value = 0;
};

TEST(WPILogWriterTest, WritesRegisteredFieldsAndSkipsMissingMessages) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<double>("temperature"),
      control_loop::MessageDescriptor::Publication<
          localization::PositionEstimateMessage>(
          "pose"),
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

TEST(WPILogWriterTest, FlushesValuesWhileWriterIsActive) {
  const auto path = LogPath();
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<std::int64_t>("count")};

  const auto has_value = [&](std::int64_t expected) {
    auto buffer = wpi::MemoryBuffer::GetFile(path.string());
    if (!buffer.has_value()) return false;
    wpi::log::DataLogReader reader(std::move(*buffer));
    if (!reader.IsValid()) return false;
    int count_entry = -1;
    for (const auto& record : reader) {
      if (record.IsStart()) {
        wpi::log::StartRecordData start;
        if (record.GetStartData(&start) && start.name == "count") {
          count_entry = start.entry;
        }
      } else if (!record.IsControl() && record.GetEntry() == count_entry) {
        std::int64_t value = 0;
        if (record.GetInteger(&value) && value == expected) return true;
      }
    }
    return false;
  };

  {
    logging::WPILogWriter writer(path.string(), publications);
    control_loop::ContextInternal first(std::chrono::steady_clock::now(),
                                        nullptr, std::stop_token{}, 1);
    first.SetMessage(
        "count", std::make_unique<control_loop::ValueMessage<std::int64_t>>(1));
    writer.Log(first);
    writer.Flush();
    EXPECT_TRUE(has_value(1));

    control_loop::ContextInternal second(std::chrono::steady_clock::now(),
                                         nullptr, std::stop_token{}, 2);
    second.SetMessage(
        "count", std::make_unique<control_loop::ValueMessage<std::int64_t>>(2));
    writer.Log(second);
    const auto deadline = std::chrono::steady_clock::now() +
                          std::chrono::seconds(3);
    while (!has_value(2) && std::chrono::steady_clock::now() < deadline) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    EXPECT_TRUE(has_value(2));
  }
  std::filesystem::remove(path);
}

TEST(WPILogWriterTest, RequiresRegistrationForClassPublications) {
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<UnregisteredSample>(
          "unregistered"),
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
      control_loop::MessageDescriptor::Publication<Pose2dSample>("location"),
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
        std::make_unique<control_loop::ValueMessage<Pose2dSample>>(
            Pose2dSample{}));
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
      } else if (name == "location/pose") {
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
