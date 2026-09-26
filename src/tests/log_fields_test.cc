#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "control_loop/context.h"
#include "control_loop/timed_node.h"
#include "localization/position.h"
#include "logging/log_registration.h"
#include "logging/wpilog_writer.h"
#include "wpilog_test_utils.h"

namespace {

struct Nested {
  double x = 0;
  LOG_FIELDS(Nested, x)
};

struct Sample {
  double a = 0;
  bool b = false;
  std::vector<double> c;
  frc::Pose3d pose;
  Nested d;
  std::vector<int> ids;
  LOG_FIELDS(Sample, a, b, c, pose, d, ids)
};

struct ArraySample {
  std::vector<Nested> ignored;
  std::optional<int> absent;
  std::vector<frc::Pose3d> poses;
  LOG_FIELDS(ArraySample, ignored, absent, poses)
};

static_assert(logging::detail::HasLogFields<Sample>);
static_assert(std::tuple_size_v<decltype(Sample::WpiLogFields())> == 6);

TEST(LogFieldsTest, IgnoresStructVectorsAndWritesNativePose3dArrays) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("cos-native-array-" + std::to_string(getpid()) + ".wpilog");
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<ArraySample>("sample")};
  {
    logging::WPILogWriter writer(path.string(), publications);
    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                          nullptr, std::stop_token{}, 1);
    ArraySample sample;
    sample.ignored = {Nested{1.0}};
    sample.absent = 2;
    sample.poses.resize(2);
    context.SetMessage(
        "sample",
        std::make_unique<control_loop::ValueMessage<ArraySample>>(sample));
    writer.Log(context);
  }

  int pose_arrays = 0;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) {
    if (name != "sample/poses") return;
    ++pose_arrays;
    EXPECT_EQ(record.GetRaw().size(), 2 * 7 * sizeof(double));
  });
  EXPECT_EQ(pose_arrays, 1);
  std::filesystem::remove(path);
}

TEST(LogFieldsTest, SameTypeInTwoSubchannelsUsesSeparateEntries) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("cos-log-fields-" + std::to_string(getpid()) + ".wpilog");
  const std::string first = "master_channel1/sub_channel1";
  const std::string second = "master_channel1/sub_channel2";
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<Sample>(first),
      control_loop::MessageDescriptor::Publication<Sample>(second)};

  {
    logging::WPILogWriter writer(path.string(), publications);
    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                          nullptr, std::stop_token{}, 1);
    Sample left;
    left.a = 1.5;
    left.b = true;
    left.c = {2.0, 3.0};
    left.d.x = 4.0;
    left.ids = {5, 6};
    Sample right;
    right.a = 11.5;
    right.d.x = 14.0;
    context.SetMessage(first, std::make_unique<control_loop::ValueMessage<Sample>>(left));
    context.SetMessage(second, std::make_unique<control_loop::ValueMessage<Sample>>(right));
    writer.Log(context);
    writer.Flush();
  }

  std::unordered_set<std::string> written;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) {
      // Pose3d registration also writes WPILib struct schemas.
      if (!name.starts_with(first + "/") &&
          !name.starts_with(second + "/")) {
        return;
      }
      written.insert(name);
      if (name == first + "/a" || name == second + "/a") {
        double value = 0;
        ASSERT_TRUE(record.GetDouble(&value));
        EXPECT_DOUBLE_EQ(value, name == first + "/a" ? 1.5 : 11.5);
      }
      if (name == first + "/d/x" || name == second + "/d/x") {
        double value = 0;
        ASSERT_TRUE(record.GetDouble(&value));
        EXPECT_DOUBLE_EQ(value, name == first + "/d/x" ? 4.0 : 14.0);
      }
      if (name == first + "/ids") {
        std::vector<std::int64_t> ids;
        ASSERT_TRUE(record.GetIntegerArray(&ids));
        EXPECT_EQ(ids, (std::vector<std::int64_t>{5, 6}));
      }
      if (name == first + "/b") {
        bool value = false;
        ASSERT_TRUE(record.GetBoolean(&value));
        EXPECT_TRUE(value);
      }
      if (name == first + "/c") {
        std::vector<double> values;
        ASSERT_TRUE(record.GetDoubleArray(&values));
        EXPECT_EQ(values, (std::vector<double>{2.0, 3.0}));
      }
  });
  EXPECT_EQ(written.size(), 12);
  EXPECT_TRUE(written.contains(first + "/pose"));
  EXPECT_TRUE(written.contains(second + "/pose"));
  std::filesystem::remove(path);
}

TEST(LogFieldsTest, ContextDestructionWritesProductionMessagesToRealLog) {
  const auto path = std::filesystem::temp_directory_path() /
                    ("cos-context-fields-" + std::to_string(getpid()) + ".wpilog");
  const std::string first = "localization/camera1/pose";
  const std::string second = "localization/camera2/pose";
  const std::string latency = "decode/camera1/latency";
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<localization::PositionEstimateMessage>(first),
      control_loop::MessageDescriptor::Publication<localization::PositionEstimateMessage>(second),
      control_loop::MessageDescriptor::Publication<control_loop::LatencyMessage>(latency)};

  auto writer = std::make_shared<logging::WPILogWriter>(path.string(), publications);
  {
    auto context = std::make_unique<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1,
        writer);
    auto left = std::make_unique<localization::PositionEstimateMessage>();
    left->tag_ids = {1, 2};
    left->num_tags = 2;
    left->distances = {3.5, 4.5};
    left->variance = 0.25;
    auto right = std::make_unique<localization::PositionEstimateMessage>();
    right->tag_ids = {9};
    right->num_tags = 1;
    right->distances = {7.5};
    right->variance = 0.75;
    context->SetMessage(first, std::move(left));
    context->SetMessage(second, std::move(right));
    context->SetMessage(
        latency,
        std::make_unique<control_loop::LatencyMessage>(
            std::chrono::duration<double>(0.012)));
  }  // ContextInternal's destructor is the only call to writer->Log().
  writer.reset();  // Close the file before reading it back.

  std::unordered_set<std::string> written;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) {
    if (!name.starts_with(first + "/") &&
        !name.starts_with(second + "/") && name != latency + "/latency") {
      return;  // WPILib also stores Pose3d schema records.
    }
    written.insert(name);
    if (name == first + "/tag_ids" || name == second + "/tag_ids") {
      std::vector<std::int64_t> ids;
      ASSERT_TRUE(record.GetIntegerArray(&ids));
      EXPECT_EQ(ids, name == first + "/tag_ids"
                         ? (std::vector<std::int64_t>{1, 2})
                         : (std::vector<std::int64_t>{9}));
    } else if (name == first + "/num_tags" ||
               name == second + "/num_tags") {
      std::int64_t count = -1;
      ASSERT_TRUE(record.GetInteger(&count));
      EXPECT_EQ(count, name == first + "/num_tags" ? 2 : 1);
    } else if (name == first + "/distances" ||
               name == second + "/distances") {
      std::vector<double> distances;
      ASSERT_TRUE(record.GetDoubleArray(&distances));
      EXPECT_EQ(distances, name == first + "/distances"
                               ? (std::vector<double>{3.5, 4.5})
                               : (std::vector<double>{7.5}));
    } else if (name == first + "/variance" ||
               name == second + "/variance") {
      double variance = 0;
      ASSERT_TRUE(record.GetDouble(&variance));
      EXPECT_DOUBLE_EQ(variance, name == first + "/variance" ? 0.25 : 0.75);
    } else if (name == latency + "/latency") {
      double seconds = 0;
      ASSERT_TRUE(record.GetDouble(&seconds));
      EXPECT_DOUBLE_EQ(seconds, 0.012);
    }
  });
  ASSERT_EQ(written,
            (std::unordered_set<std::string>{
                first + "/tag_ids", first + "/num_tags", first + "/pose", first + "/distances",
                first + "/variance", second + "/tag_ids", second + "/num_tags", second + "/pose",
                second + "/distances", second + "/variance",
                latency + "/latency"}));
  // An explicit destination lets a device run retain the verified file.
  if (const char* proof_path = std::getenv("COS_WPILOG_PROOF_PATH")) {
    ASSERT_TRUE(std::filesystem::copy_file(
        path, proof_path, std::filesystem::copy_options::overwrite_existing));
  }
  std::filesystem::remove(path);
}

}  // namespace
