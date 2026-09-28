#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <stop_token>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>
#include <unistd.h>

#include "control_loop/context.h"
#include "control_loop/timed_node.h"
#include "localization/position.h"
#include "localization/solver_common.h"
#include "logging/log_registration.h"
#include "logging/wpilog_writer.h"
#include "wpilog_test_utils.h"

namespace {

struct Nested {
  double x = 0;
  LOG_FIELDS(Nested, x)
};
enum class OptionalEnum { populated = 1 };

template <typename T>
struct OptionalSample {
  std::optional<T> value;
  LOG_FIELDS(OptionalSample, value)
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

static_assert(requires { Sample::WpiLogFields(); });
static_assert(std::tuple_size_v<decltype(Sample::WpiLogFields())> == 6);

TEST(LogFieldsTest, WritesNativeArraysAndOptionalPrimitivesAndPose3d) {
  const auto path =
      std::filesystem::temp_directory_path() /
      ("cos-native-array-" + std::to_string(getpid()) + ".wpilog");
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<ArraySample>("sample")};
  {
    logging::WPILogWriter writer(path.string(), publications);
    control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                          nullptr, std::stop_token{}, 1);
    ArraySample sample;
    sample.ignored.resize(1);
    sample.absent = 2;
    sample.poses.resize(2);
    context.SetMessage(
        "sample",
        std::make_unique<control_loop::ValueMessage<ArraySample>>(sample));
    writer.Log(context);
  }

  int pose_arrays = 0;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) {
    EXPECT_FALSE(name.starts_with("sample/ignored"));
    if (name != "sample/poses")
      return;
    ++pose_arrays;
    EXPECT_EQ(record.GetRaw().size(), 2 * 7 * sizeof(double));
  });
  EXPECT_EQ(pose_arrays, 1);
  std::filesystem::remove(path);

  const auto check_optional = [&]<typename T>() {
    T populated{};
    if constexpr (std::is_same_v<T, frc::Pose3d>) {
      populated = frc::Pose3d{units::meter_t{1}, units::meter_t{2},
                              units::meter_t{3}, frc::Rotation3d{}};
    } else if constexpr (std::is_same_v<T, std::string>)
      populated = "present";
    else
      populated = static_cast<T>(1);
    const std::vector optional_publications{
        control_loop::MessageDescriptor::Publication<OptionalSample<T>>(
            "sample")};
    {
      logging::WPILogWriter writer(path.string(), optional_publications);
      for (bool present : {false, true, false}) {
        control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                              nullptr, std::stop_token{}, 1);
        OptionalSample<T> sample;
        if (present)
          sample.value = populated;
        context.SetMessage(
            "sample",
            std::make_unique<control_loop::ValueMessage<OptionalSample<T>>>(
                sample));
        writer.Log(context);
      }
    }
    int flags = 0, values = 0;
    wpilog_test::VisitLogValues(path, [&](const auto& name,
                                          const auto& record) {
      if (name == "sample/value_present") {
        bool present = false;
        ASSERT_TRUE(record.GetBoolean(&present));
        EXPECT_EQ(present, flags++ == 1);
      } else if (name == "sample/value") {
        const T expected = values++ == 1 ? populated : T{};
        if constexpr (std::is_same_v<T, frc::Pose3d>) {
          ASSERT_EQ(record.GetRaw().size(), 7 * sizeof(double));
          EXPECT_EQ(wpi::UnpackStruct<frc::Pose3d>(record.GetRaw()), expected);
        } else if constexpr (std::is_same_v<T, std::string>) {
          std::string_view value;
          ASSERT_TRUE(record.GetString(&value));
          EXPECT_EQ(value, expected);
        } else if constexpr (std::is_same_v<T, bool>) {
          bool value = false;
          ASSERT_TRUE(record.GetBoolean(&value));
          EXPECT_EQ(value, expected);
        } else if constexpr (std::is_same_v<T, float>) {
          float value = 0;
          ASSERT_TRUE(record.GetFloat(&value));
          EXPECT_EQ(value, expected);
        } else if constexpr (std::is_floating_point_v<T>) {
          double value = 0;
          ASSERT_TRUE(record.GetDouble(&value));
          EXPECT_EQ(value, static_cast<double>(expected));
        } else {
          std::int64_t value = 0;
          ASSERT_TRUE(record.GetInteger(&value));
          EXPECT_EQ(value, static_cast<std::int64_t>(expected));
        }
      }
    });
    EXPECT_EQ(flags, 3);
    EXPECT_EQ(values, 3);
    std::filesystem::remove(path);
  };
  std::apply(
      [&](auto... values) {
        (check_optional.template operator()<decltype(values)>(), ...);
      },
      std::tuple<bool, char, signed char, unsigned char, short, unsigned short,
                 int, unsigned int, long, unsigned long, long long,
                 unsigned long long, wchar_t, char8_t, char16_t, char32_t,
                 float, double, long double, OptionalEnum, std::string,
                 frc::Pose3d>{});

  const std::vector nested_publications{
      control_loop::MessageDescriptor::Publication<
          localization::AmbiguousEstimate>("estimate"),
      control_loop::MessageDescriptor::Publication<
          localization::AmbiguousEstimateMessage>("batch")};
  const frc::Pose3d populated_pose{units::meter_t{1}, units::meter_t{2},
                                   units::meter_t{3}, frc::Rotation3d{}};
  {
    logging::WPILogWriter writer(path.string(), nested_publications);
    for (bool present : {false, true, false}) {
      control_loop::ContextInternal context(std::chrono::steady_clock::now(),
                                            nullptr, std::stop_token{}, 1);
      localization::AmbiguousEstimate estimate;
      estimate.pos1.variance = 2;
      if (present) {
        estimate.pos2.emplace();
        estimate.pos2->tag_ids = {9};
        estimate.pos2->distances = {5};
        estimate.pos2->pose = populated_pose;
        estimate.pos2->variance = 3;
        estimate.pos2->distance = 4;
      }
      context.SetMessage(
          "estimate",
          std::make_unique<
              control_loop::ValueMessage<localization::AmbiguousEstimate>>(
              estimate));
      context.SetMessage(
          "batch",
          std::make_unique<localization::AmbiguousEstimateMessage>(estimate));
      writer.Log(context);
    }
  }
  std::unordered_map<std::string, int> counts;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) {
    const std::string field_name =
        name.starts_with("batch/") ? name.substr(6) : name;
    if (!field_name.starts_with("estimate/"))
      return;
    const bool present = counts[name]++ == 1;
    if (field_name == "estimate/pos2_present") {
      bool value = false;
      ASSERT_TRUE(record.GetBoolean(&value));
      EXPECT_EQ(value, present);
    } else if (field_name == "estimate/pos2/pose") {
      ASSERT_EQ(record.GetRaw().size(), 7 * sizeof(double));
      EXPECT_EQ(wpi::UnpackStruct<frc::Pose3d>(record.GetRaw()),
                present ? populated_pose : frc::Pose3d{});
    } else if (field_name == "estimate/pos2/variance" ||
               field_name == "estimate/pos2/distance" ||
               field_name == "estimate/pos1/variance") {
      double value = 0;
      ASSERT_TRUE(record.GetDouble(&value));
      EXPECT_EQ(value,
                field_name == "estimate/pos1/variance"
                    ? 2
                    : (present ? (name.ends_with("variance") ? 3 : 4) : 0));
    } else if (field_name == "estimate/pos2/tag_ids") {
      std::vector<std::int64_t> values;
      ASSERT_TRUE(record.GetIntegerArray(&values));
      EXPECT_EQ(values, present ? std::vector<std::int64_t>{9}
                                : std::vector<std::int64_t>{});
    } else if (field_name == "estimate/pos2/distances") {
      std::vector<double> values;
      ASSERT_TRUE(record.GetDoubleArray(&values));
      EXPECT_EQ(values,
                present ? std::vector<double>{5} : std::vector<double>{});
    }
  });
  EXPECT_EQ(counts.size(), 22);
  for (const auto& [name, count] : counts)
    EXPECT_EQ(count, 3) << name;
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
    context.SetMessage(
        first, std::make_unique<control_loop::ValueMessage<Sample>>(left));
    context.SetMessage(
        second, std::make_unique<control_loop::ValueMessage<Sample>>(right));
    writer.Log(context);
    writer.Flush();
  }

  std::unordered_set<std::string> written;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) {
    // Pose3d registration also writes WPILib struct schemas.
    if (!name.starts_with(first + "/") && !name.starts_with(second + "/")) {
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
  const auto path =
      std::filesystem::temp_directory_path() /
      ("cos-context-fields-" + std::to_string(getpid()) + ".wpilog");
  const std::string first = "localization/camera1/pose";
  const std::string second = "localization/camera2/pose";
  const std::string latency = "decode/camera1/latency";
  const std::vector publications{
      control_loop::MessageDescriptor::Publication<
          localization::PositionEstimateMessage>(first),
      control_loop::MessageDescriptor::Publication<
          localization::PositionEstimateMessage>(second),
      control_loop::MessageDescriptor::Publication<
          control_loop::LatencyMessage>(latency)};

  auto writer =
      std::make_shared<logging::WPILogWriter>(path.string(), publications);
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
    context->SetMessage(latency, std::make_unique<control_loop::LatencyMessage>(
                                     std::chrono::duration<double>(0.012)));
  }  // ContextInternal's destructor is the only call to writer->Log().
  writer.reset();  // Close the file before reading it back.

  std::unordered_set<std::string> written;
  wpilog_test::VisitLogValues(path, [&](const auto& name, const auto& record) {
    if (!name.starts_with(first + "/") && !name.starts_with(second + "/") &&
        name != latency + "/latency") {
      return;  // WPILib also stores Pose3d schema records.
    }
    written.insert(name);
    if (name == first + "/tag_ids" || name == second + "/tag_ids") {
      std::vector<std::int64_t> ids;
      ASSERT_TRUE(record.GetIntegerArray(&ids));
      EXPECT_EQ(ids, name == first + "/tag_ids"
                         ? (std::vector<std::int64_t>{1, 2})
                         : (std::vector<std::int64_t>{9}));
    } else if (name == first + "/num_tags" || name == second + "/num_tags") {
      std::int64_t count = -1;
      ASSERT_TRUE(record.GetInteger(&count));
      EXPECT_EQ(count, name == first + "/num_tags" ? 2 : 1);
    } else if (name == first + "/distances" || name == second + "/distances") {
      std::vector<double> distances;
      ASSERT_TRUE(record.GetDoubleArray(&distances));
      EXPECT_EQ(distances, name == first + "/distances"
                               ? (std::vector<double>{3.5, 4.5})
                               : (std::vector<double>{7.5}));
    } else if (name == first + "/variance" || name == second + "/variance") {
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
                first + "/tag_ids", first + "/num_tags", first + "/pose",
                first + "/distances", first + "/variance", second + "/tag_ids",
                second + "/num_tags", second + "/pose", second + "/distances",
                second + "/variance", latency + "/latency"}));
  // An explicit destination lets a device run retain the verified file.
  if (const char* proof_path = std::getenv("COS_WPILOG_PROOF_PATH")) {
    ASSERT_TRUE(std::filesystem::copy_file(
        path, proof_path, std::filesystem::copy_options::overwrite_existing));
  }
  std::filesystem::remove(path);
}

}  // namespace
