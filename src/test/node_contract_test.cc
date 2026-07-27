#include <atomic>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <typeindex>

#include <gtest/gtest.h>

#include "camera/jpeg_disk_camera.h"
#include "control_loop/context.h"
#include "control_loop/thread_pool.h"
#include "logging/jpeg_buffer_log_node.h"
#include "streamer/jpeg_buffer_streamer_node.h"

namespace {

auto MakeContext() -> control_loop::Context {
  return std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{});
}

TEST(JpegDiskCameraTest, PublishesJpegBuffers) {
  const auto directory =
      std::filesystem::temp_directory_path() / "cos-empty-camera-test";
  std::filesystem::create_directories(directory);
  camera::JpegDiskCamera node(directory.string(), "jpeg", false);

  ASSERT_TRUE(node.GetDependencies().empty());
  ASSERT_EQ(node.GetPublications().size(), 1U);
  EXPECT_TRUE(node.GetPublications()[0].GetTypes().contains(
      std::type_index(typeid(camera::JpegBuffer))));

  const control_loop::Context context = MakeContext();
  node.CreateCallback()(context);
  EXPECT_EQ(context->GetMessage<camera::JpegBuffer>("jpeg"), nullptr);

  std::filesystem::remove_all(directory);
}

TEST(JpegBufferStreamerNodeTest, ConsumesJpegBuffers) {
  streamer::JpegBufferStreamerNode node("jpeg", "/test", 0);

  ASSERT_EQ(node.GetDependencies().size(), 1U);
  EXPECT_EQ(node.GetDependencies()[0].GetChannel(), "jpeg");
  EXPECT_TRUE(node.GetDependencies()[0].GetTypes().contains(
      std::type_index(typeid(camera::JpegBuffer))));
  EXPECT_TRUE(node.GetPublications().empty());
}

TEST(JpegBufferLogNodeTest, ConsumesJpegBuffers) {
  control_loop::ThreadPool thread_pool(1);
  const auto directory =
      std::filesystem::temp_directory_path() / "cos-jpeg-log-test";
  std::filesystem::create_directories(directory);
  logging::JpegBufferLogNode node("jpeg", directory.string(), thread_pool);

  ASSERT_EQ(node.GetDependencies().size(), 1U);
  EXPECT_EQ(node.GetDependencies()[0].GetChannel(), "jpeg");
  EXPECT_TRUE(node.GetDependencies()[0].GetTypes().contains(
      std::type_index(typeid(camera::JpegBuffer))));
  EXPECT_TRUE(node.GetPublications().empty());

  const control_loop::Context context = MakeContext();
  std::atomic<int> callback_count = 0;
  node.RegisterCallback(
      [&callback_count](const control_loop::Context&) -> void {
        ++callback_count;
      });
  auto jpeg = std::make_unique<camera::JpegBuffer>(4, 7.25);
  std::memcpy(jpeg->ptr, "jpeg", 4);
  context->SetMessage("jpeg", std::move(jpeg));
  node.CreateCallback()(context);
  thread_pool.Shutdown();

  std::ifstream output(directory / "7.250000.jpg", std::ios::binary);
  EXPECT_TRUE(output.is_open());
  std::string contents((std::istreambuf_iterator<char>(output)),
                       std::istreambuf_iterator<char>());
  EXPECT_EQ(contents, "jpeg");
  EXPECT_EQ(callback_count.load(), 1);
  std::filesystem::remove_all(directory);
}

}  // namespace
