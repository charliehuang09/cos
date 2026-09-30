#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <stop_token>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "camera/nvjpeg_decode_node.h"
#include "control_loop/context.h"
#include "control_loop/message.h"
#include "control_loop/node.h"
#include "gamepiece/gamepiece_control_loop.h"

using namespace std::chrono_literals;

namespace {

class TestMessage final : public control_loop::IMessage {
 public:
  auto GetType() -> const std::type_info& override {
    return typeid(TestMessage);
  }
  auto GetSize() -> size_t override { return sizeof(*this); }
};

struct TestContext {
  TestContext()
      : context(new control_loop::ContextInternal(
            std::chrono::steady_clock::now(), nullptr, stop_source.get_token(),
            &destructed)) {}

  std::atomic<bool> destructed = false;
  std::stop_source stop_source;
  control_loop::Context context;
};

class FakeDecoderNode final : public control_loop::INode {
 public:
  explicit FakeDecoderNode(std::string_view channel)
      : publications_({{channel, typeid(camera::DecodedJpegBuffer)}}) {}

  void Emit(const control_loop::Context& context) {
    for (const auto& callback : callbacks_) {
      callback(context);
    }
  }

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [](const control_loop::Context&) {};
  }

  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback)
      override {
    callbacks_.push_back(callback);
  }

  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return dependencies_;
  }

  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return publications_;
  }

 private:
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

class FakeGamepieceNode final : public control_loop::INode {
 public:
  using Observer =
      std::function<void(const control_loop::Context&,
                         const std::shared_ptr<camera::DecodedJpegBuffer>&)>;

  FakeGamepieceNode(std::string_view channel, Observer observer)
      : channel_(channel),
        observer_(std::move(observer)),
        dependencies_({{channel_, typeid(camera::DecodedJpegBuffer)}}) {}

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override {
    return [this](const control_loop::Context& context) {
      auto frame =
          context->GetSharedMessage<camera::DecodedJpegBuffer>(channel_);
      if (frame != nullptr) {
        observer_(context, frame);
      }
      for (const auto& callback : callbacks_) {
        callback(context);
      }
    };
  }

  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback)
      override {
    callbacks_.push_back(callback);
  }

  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return dependencies_;
  }

  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override {
    return publications_;
  }

 private:
  std::string channel_;
  Observer observer_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
};

auto EmitFrame(FakeDecoderNode& decoder, std::string_view channel,
               double timestamp)
    -> std::weak_ptr<camera::DecodedJpegBuffer> {
  TestContext localization;
  auto frame = std::make_shared<camera::DecodedJpegBuffer>();
  frame->timestamp = timestamp;
  std::weak_ptr<camera::DecodedJpegBuffer> weak_frame = frame;
  localization.context->SetMessage(channel, frame);
  localization.context->SetMessage("localization-only",
                                   std::make_unique<TestMessage>());
  decoder.Emit(localization.context);
  frame.reset();
  localization.context.reset();
  EXPECT_TRUE(localization.destructed.load());
  return weak_frame;
}

TEST(ContextSharedMessageTest, SharedMessageOutlivesContext) {
  std::shared_ptr<TestMessage> retained;
  std::weak_ptr<TestMessage> weak_message;
  TestContext owner;
  {
    auto message = std::make_shared<TestMessage>();
    weak_message = message;
    owner.context->SetMessage("shared", message);
    retained = owner.context->GetSharedMessage<TestMessage>("shared");
    ASSERT_EQ(retained, message);
  }

  owner.context.reset();
  EXPECT_TRUE(owner.destructed.load());
  EXPECT_FALSE(weak_message.expired());
  retained.reset();
  EXPECT_TRUE(weak_message.expired());
}

TEST(GamepieceControlLoopTest, KeepsInFlightFrameAndConsumesLatestFrame) {
  constexpr std::string_view kChannel = "decoded/front";
  auto decoder = std::make_shared<FakeDecoderNode>(kChannel);

  std::mutex mutex;
  std::condition_variable condition;
  std::vector<double> observed_timestamps;
  bool first_frame_entered = false;
  bool release_first_frame = false;
  bool copied_localization_message = false;

  auto consumer = std::make_shared<FakeGamepieceNode>(
      kChannel,
      [&](const control_loop::Context& context,
          const std::shared_ptr<camera::DecodedJpegBuffer>& frame) {
        std::unique_lock lock(mutex);
        copied_localization_message |=
            context->GetMessage<TestMessage>("localization-only") != nullptr;
        if (observed_timestamps.size() < 32U) {
          observed_timestamps.push_back(frame->timestamp);
        }
        if (frame->timestamp == 1.0 && !first_frame_entered) {
          first_frame_entered = true;
          condition.notify_all();
          condition.wait(lock, [&] { return release_first_frame; });
        }
        condition.notify_all();
      });

  gamepiece::GamepieceControlLoop loop;
  loop.RegisterDecodedFrameSource(decoder, kChannel);
  loop.RegisterNode(consumer);

  std::weak_ptr<camera::DecodedJpegBuffer> first_frame =
      EmitFrame(*decoder, kChannel, 1.0);
  loop.Start();
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 2s, [&] {
      return first_frame_entered;
    }));
  }

  const std::weak_ptr<camera::DecodedJpegBuffer> second_frame =
      EmitFrame(*decoder, kChannel, 2.0);
  const std::weak_ptr<camera::DecodedJpegBuffer> third_frame =
      EmitFrame(*decoder, kChannel, 3.0);
  EXPECT_FALSE(first_frame.expired());
  EXPECT_TRUE(second_frame.expired());
  EXPECT_FALSE(third_frame.expired());

  {
    std::lock_guard lock(mutex);
    release_first_frame = true;
  }
  condition.notify_all();
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 2s, [&] {
      return std::ranges::find(observed_timestamps, 3.0) !=
             observed_timestamps.end();
    }));
  }
  loop.Stop();

  EXPECT_TRUE(first_frame.expired());
  EXPECT_TRUE(third_frame.expired());
  EXPECT_FALSE(copied_localization_message);
  EXPECT_EQ(std::ranges::find(observed_timestamps, 2.0),
            observed_timestamps.end());
}

TEST(GamepieceControlLoopTest, KeepsCameraChannelsIndependent) {
  constexpr std::string_view kFrontChannel = "decoded/front";
  constexpr std::string_view kRearChannel = "decoded/rear";
  auto front_decoder = std::make_shared<FakeDecoderNode>(kFrontChannel);
  auto rear_decoder = std::make_shared<FakeDecoderNode>(kRearChannel);

  std::mutex mutex;
  std::condition_variable condition;
  bool saw_front = false;
  bool saw_rear = false;
  bool mixed_channels = false;

  auto front_consumer = std::make_shared<FakeGamepieceNode>(
      kFrontChannel,
      [&](const control_loop::Context& context,
          const std::shared_ptr<camera::DecodedJpegBuffer>& frame) {
        std::lock_guard lock(mutex);
        saw_front |= frame->timestamp == 11.0;
        mixed_channels |=
            context
                ->GetSharedMessage<camera::DecodedJpegBuffer>(kRearChannel) !=
            nullptr;
        condition.notify_all();
      });
  auto rear_consumer = std::make_shared<FakeGamepieceNode>(
      kRearChannel,
      [&](const control_loop::Context&,
          const std::shared_ptr<camera::DecodedJpegBuffer>& frame) {
        std::lock_guard lock(mutex);
        saw_rear |= frame->timestamp == 22.0;
        condition.notify_all();
      });

  gamepiece::GamepieceControlLoop loop;
  loop.RegisterDecodedFrameSource(front_decoder, kFrontChannel);
  loop.RegisterDecodedFrameSource(rear_decoder, kRearChannel);
  loop.RegisterNode(front_consumer);
  loop.RegisterNode(rear_consumer);
  EmitFrame(*front_decoder, kFrontChannel, 11.0);
  EmitFrame(*rear_decoder, kRearChannel, 22.0);

  loop.Start();
  {
    std::unique_lock lock(mutex);
    ASSERT_TRUE(condition.wait_for(lock, 2s,
                                   [&] { return saw_front && saw_rear; }));
  }
  loop.Stop();

  EXPECT_FALSE(mixed_channels);
}

TEST(GamepieceControlLoopTest, StopsBeforeAnyFrameArrives) {
  constexpr std::string_view kChannel = "decoded/front";
  auto decoder = std::make_shared<FakeDecoderNode>(kChannel);
  auto consumer = std::make_shared<FakeGamepieceNode>(
      kChannel,
      [](const control_loop::Context&,
         const std::shared_ptr<camera::DecodedJpegBuffer>&) {
        FAIL() << "A gamepiece node ran without a decoded frame";
      });

  gamepiece::GamepieceControlLoop loop;
  loop.RegisterDecodedFrameSource(decoder, kChannel);
  loop.RegisterNode(consumer);
  loop.Start();
  std::this_thread::sleep_for(10ms);

  const auto stop_start = std::chrono::steady_clock::now();
  loop.Stop();
  EXPECT_LT(std::chrono::steady_clock::now() - stop_start, 500ms);
}

}  // namespace
