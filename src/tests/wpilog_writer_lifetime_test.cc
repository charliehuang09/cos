#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include <wpi/DataLogReader.h>
#include <wpi/MemoryBuffer.h>

#include "camera/jpeg_buffer.h"
#include "control_loop/context.h"
#include "control_loop/control_loop.h"
#include "logging/latency_log.h"
#include "logging/wpilog_writer.h"

namespace {

class LatencyPublisher final : public control_loop::INode {
 public:
  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [](const control_loop::Context& context) {
      context->SetMessage(
          "latency", std::make_unique<control_loop::LatencyMessage>(
                         std::chrono::duration<double>{0.25}));
    };
  }

  auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return dependencies_;
  }

  auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return publications_;
  }

  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>&) override {}

 private:
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_ = {
      control_loop::MessageDescriptor::For<control_loop::LatencyMessage>(
          "latency")};
};

}  // namespace

TEST(WPILogWriter, ContextRetainsWriterUntilLastMessage) {
  constexpr std::string_view path = "/tmp/cos_wpilog_writer_lifetime_test.wpilog";
  auto writer = std::make_shared<logging::WPILogWriter>(std::string(path));
  writer->Configure({control_loop::MessageDescriptor::For<
      control_loop::LatencyMessage>("latency"),
      control_loop::MessageDescriptor::For<camera::JpegBuffer>("frame")});

  {
    auto context = std::make_shared<control_loop::ContextInternal>(
        std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 1,
        writer);
    context->SetMessage("frame",
                        std::make_unique<camera::JpegBuffer>(0, 123.456));
    context->SetMessage(
        "latency", std::make_unique<control_loop::LatencyMessage>(
                       std::chrono::duration<double>{0.25}));
    writer.reset();
  }

  auto buffer = wpi::MemoryBuffer::GetFile(path);
  ASSERT_TRUE(buffer.has_value());
  wpi::log::DataLogReader reader(std::move(buffer.value()));
  ASSERT_TRUE(reader.IsValid());
  size_t samples = 0;
  for (const auto& record : reader) {
    if (!record.IsControl()) {
      double value = 0.0;
      ASSERT_TRUE(record.GetDouble(&value));
      EXPECT_DOUBLE_EQ(value, 0.25);
      EXPECT_EQ(record.GetTimestamp(), 123'456'000);
      ++samples;
    }
  }
  EXPECT_EQ(samples, 1U);
}

TEST(WPILogWriter, ControlLoopWritesPublishedMessages) {
  constexpr std::string_view path = "/tmp/cos_wpilog_control_loop_test.wpilog";
  auto writer = std::make_shared<logging::WPILogWriter>(std::string(path));
  control_loop::ControlLoop loop(std::chrono::milliseconds(1));
  std::promise<void> first_context;
  std::once_flag first_context_once;
  std::atomic<bool> context_has_loop = false;
  auto ready = first_context.get_future();
  loop.RegisterDependancyNode(std::make_shared<LatencyPublisher>());
  loop.RegisterCallback([&](const control_loop::Context& context) {
    context_has_loop = context->control_loop == &loop;
    std::call_once(first_context_once, [&] { first_context.set_value(); });
  });
  loop.SetWPILogWriter(writer);
  loop.Start();
  const auto status = ready.wait_for(std::chrono::seconds(5));
  loop.Stop();
  writer->Close();
  ASSERT_EQ(status, std::future_status::ready);
  EXPECT_TRUE(context_has_loop);

  auto buffer = wpi::MemoryBuffer::GetFile(path);
  ASSERT_TRUE(buffer.has_value());
  wpi::log::DataLogReader reader(std::move(buffer.value()));
  ASSERT_TRUE(reader.IsValid());
  size_t samples = 0;
  for (const auto& record : reader) {
    if (!record.IsControl()) {
      double value = 0.0;
      ASSERT_TRUE(record.GetDouble(&value));
      EXPECT_DOUBLE_EQ(value, 0.25);
      ++samples;
    }
  }
  EXPECT_GT(samples, 0U);
}
