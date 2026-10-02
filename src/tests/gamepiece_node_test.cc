#include <chrono>
#include <future>
#include <memory>
#include <stop_token>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "camera/nvjpeg_decode_node.h"
#include "control_loop/context.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/gamepiece_node.h"

using namespace std::chrono_literals;

namespace {

constexpr std::string_view kDecodedChannel = "gamepiece/decoded";
constexpr std::string_view kDetectionsChannel = "gamepiece/detections";

class BlockingDetector final : public gamepiece::ObjectDetector {
 public:
  explicit BlockingDetector(std::shared_future<void> released)
      : released_(std::move(released)) {}

  auto Detect(const cv::cuda::GpuMat&) -> std::vector<gamepiece::LabeledBoundingBox>
      override {
    started.set_value();
    released_.wait();
    return {};
  }

  std::promise<void> started;

 private:
  std::shared_future<void> released_;
};

TEST(GamepieceNodeTest, SubmitsDetectionAndPublishesFromWorker) {
  control_loop::ThreadPool thread_pool(1);
  std::promise<void> release_detector;
  auto detector = std::make_unique<BlockingDetector>(
      release_detector.get_future().share());
  auto started = detector->started.get_future();
  const nlohmann::json intrinsics = {
      {"cx", 0.0}, {"cy", 0.0}, {"fx", 1.0}, {"fy", 1.0}};
  const nlohmann::json extrinsics = {
      {"translation_x", 0.0}, {"translation_y", 0.0},
      {"translation_z", 1.0}, {"rotation_x", 0.0},
      {"rotation_y", 0.0}, {"rotation_z", 0.0}};
  gamepiece::GamepieceNode node(std::move(detector), kDecodedChannel,
                                kDetectionsChannel, intrinsics, extrinsics,
                                thread_pool);

  auto context = std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 0);
  static unsigned char pixel = 0;
  auto frame = std::make_shared<camera::DecodedJpegBuffer>();
  frame->width = 1;
  frame->height = 1;
  frame->stride = 1;
  frame->output_format = NVJPEG_OUTPUT_Y;
  frame->destination.channel[0] = &pixel;
  context->SetMessage(kDecodedChannel, std::move(frame));

  std::promise<control_loop::Context> published;
  auto publication = published.get_future();
  node.RegisterCallback([&](const control_loop::Context& published_context) {
    published.set_value(published_context);
  });

  std::promise<void> submitted;
  auto submission = submitted.get_future();
  std::jthread submitter([&] {
    node.CreateCallback()(context);
    submitted.set_value();
  });

  const bool detection_started =
      started.wait_for(2s) == std::future_status::ready;
  const bool returned_while_detection_blocked =
      submission.wait_for(500ms) == std::future_status::ready;
  release_detector.set_value();
  submitter.join();
  thread_pool.Shutdown();

  EXPECT_TRUE(detection_started);
  EXPECT_TRUE(returned_while_detection_blocked);
  ASSERT_EQ(publication.wait_for(2s), std::future_status::ready);
  EXPECT_EQ(publication.get(), context);
  EXPECT_NE(context->GetMessage<gamepiece::GamepieceDetections>(
                kDetectionsChannel),
            nullptr);
}

}  // namespace
