#include <atomic>
#include <chrono>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <typeindex>

#include <gtest/gtest.h>
#include <nvjpeg.h>

#include "camera/cpu_decode_node.h"
#include "camera/nvjpeg_decode_node.h"
#include "camera/nvjpeg_fd_decode_node.h"
#include "control_loop/context.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"

#ifndef JPEG_TEST_IMAGE
#error "JPEG_TEST_IMAGE must point to a JPEG fixture"
#endif

namespace {
using camera::DecodedImageBuffer;
using camera::DecodedJpegBuffer;
using std::string_view;

using Decoder = control_loop::INode;
using Factory = std::function<std::unique_ptr<Decoder>(
    std::string_view, std::string_view, control_loop::ThreadPool&)>;

struct DecoderSpec {
  std::string name;
  std::type_index output_type;
  Factory make;
};

auto MakeContext() -> control_loop::Context {
  return std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{}, 0);
}

class JpegDecoderNodeTest : public testing::TestWithParam<DecoderSpec> {};

TEST_P(JpegDecoderNodeTest, DescribesItsJpegInputAndOutput) {
  control_loop::ThreadPool thread_pool(1);
  auto node = GetParam().make("jpeg", "decoded", thread_pool);

  ASSERT_EQ(node->GetDependencies().size(), 1U);
  EXPECT_EQ(node->GetDependencies()[0].GetChannel(), "jpeg");
  EXPECT_TRUE(node->GetDependencies()[0].GetTypes().contains(
      std::type_index(typeid(camera::JpegBuffer))));
  ASSERT_EQ(node->GetPublications().size(), 1U);
  EXPECT_EQ(node->GetPublications()[0].GetChannel(), "decoded");
  EXPECT_TRUE(
      node->GetPublications()[0].GetTypes().contains(GetParam().output_type));
}

TEST_P(JpegDecoderNodeTest, NotifiesCallbacksWhenInputIsAbsent) {
  control_loop::ThreadPool thread_pool(1);
  auto node = GetParam().make("jpeg", "decoded", thread_pool);
  std::atomic<int> callback_count = 0;
  node->RegisterCallback(
      [&callback_count](const control_loop::Context&) -> void {
        ++callback_count;
      });

  const control_loop::Context context = MakeContext();
  node->CreateCallback()(context);

  EXPECT_EQ(callback_count.load(), 1);
}

TEST(CpuJpegDecodeNodeTest, DecodesARealJpeg) {
  std::ifstream input(JPEG_TEST_IMAGE, std::ios::binary | std::ios::ate);
  ASSERT_TRUE(input.is_open());
  const std::streamsize size = input.tellg();
  ASSERT_GT(size, 0);
  input.seekg(0);

  control_loop::ThreadPool thread_pool(1);
  camera::CpuJpegDecodeNode node("jpeg", "decoded", thread_pool);
  const control_loop::Context context = MakeContext();
  auto jpeg = std::make_unique<camera::JpegBuffer>(size, 42.5);
  ASSERT_TRUE(input.read(reinterpret_cast<char*>(jpeg->ptr), size));
  context->SetMessage("jpeg", std::move(jpeg));

  node.CreateCallback()(context);
  thread_pool.Shutdown();

  const auto* decoded =
      context->GetMessage<camera::DecodedImageBuffer>("decoded");
  ASSERT_NE(decoded, nullptr);
  EXPECT_GT(decoded->width, 0);
  EXPECT_GT(decoded->height, 0);
  EXPECT_EQ(decoded->stride, static_cast<size_t>(decoded->width));
  EXPECT_EQ(decoded->data.size(),
            decoded->stride * static_cast<size_t>(decoded->height));
  EXPECT_DOUBLE_EQ(decoded->timestamp, 42.5);
}

INSTANTIATE_TEST_SUITE_P(
    Implementations, JpegDecoderNodeTest,
    testing::Values(
        DecoderSpec{"cpu", std::type_index(typeid(DecodedImageBuffer)),
                    [](std::string_view input, string_view output,
                       control_loop::ThreadPool& pool)
                        -> std::unique_ptr<camera::CpuJpegDecodeNode> {
                      return std::make_unique<camera::CpuJpegDecodeNode>(
                          input, output, pool);
                    }},
        DecoderSpec{"nvjpeg", std::type_index(typeid(DecodedJpegBuffer)),
                    [](std::string_view input, string_view output,
                       control_loop::ThreadPool& pool)
                        -> std::unique_ptr<camera::NvjpegDecodeNode> {
                      return std::make_unique<camera::NvjpegDecodeNode>(
                          input, output, NVJPEG_OUTPUT_Y, pool);
                    }},
        DecoderSpec{"nvjpeg_fd",
                    std::type_index(typeid(camera::DecodedJpegFdBuffer)),
                    [](std::string_view input, string_view output,
                       control_loop::ThreadPool& pool)
                        -> std::unique_ptr<camera::NvjpegFdDecodeNode> {
                      return std::make_unique<camera::NvjpegFdDecodeNode>(
                          input, output, pool);
                    }}),
    [](const testing::TestParamInfo<DecoderSpec>& info) -> std::string {
      return info.param.name;
    });

}  // namespace
