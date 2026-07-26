#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <typeindex>
#include <vector>

#include <gtest/gtest.h>

#include "apriltag/tag_detections.h"
#include "control_loop/context.h"
#include "control_loop/node.h"
#include "localization/joint_solver_node.h"
#include "localization/multi_tag_solver_node.h"
#include "localization/square_solver_node.h"
#include "localization/unambiguous_solver_node.h"

#ifndef POSE_TEST_CONSTANTS
#error "POSE_TEST_CONSTANTS must point to the camera constants fixture"
#endif

namespace {

using Solver = control_loop::INode;
using Factory = std::function<std::unique_ptr<Solver>(
    const std::vector<localization::camera_constant_t>&)>;

struct SolverSpec {
  std::string name;
  size_t dependency_count;
  std::type_index publication_type;
  Factory make;
};

auto CameraConstants() -> std::vector<localization::camera_constant_t> {
  return {{"front", std::string(POSE_TEST_CONSTANTS) + "/front_intrinsics.json",
           std::string(POSE_TEST_CONSTANTS) + "/front_extrinsics.json", ""}};
}

auto MakeContext(std::atomic<bool>& destructed) -> control_loop::Context {
  return std::make_shared<control_loop::ContextInternal>(
      std::chrono::steady_clock::now(), nullptr, std::stop_token{},
      &destructed);
}

class PoseSolverNodeTest : public testing::TestWithParam<SolverSpec> {};

TEST_P(PoseSolverNodeTest, DescribesItsInputsAndOutput) {
  auto constants = CameraConstants();
  auto node = GetParam().make(constants);

  EXPECT_EQ(node->GetDependencies().size(), GetParam().dependency_count);
  for (const auto& dependency : node->GetDependencies()) {
    EXPECT_TRUE(dependency.GetTypes().contains(
        std::type_index(typeid(apriltag::TagDetections))));
  }
  ASSERT_EQ(node->GetPublications().size(), 1U);
  EXPECT_TRUE(node->GetPublications()[0].GetTypes().contains(
      GetParam().publication_type));
}

TEST_P(PoseSolverNodeTest, EmptyInputDoesNotPublishAPose) {
  auto constants = CameraConstants();
  auto node = GetParam().make(constants);
  std::atomic<bool> destructed = false;
  const control_loop::Context context = MakeContext(destructed);
  node->CreateCallback()(context);

  EXPECT_EQ(context->GetMessage<localization::PositionEstimate>(
                node->GetPublications()[0].GetChannel()),
            nullptr);
}

INSTANTIATE_TEST_SUITE_P(
    Implementations, PoseSolverNodeTest,
    testing::Values(
        SolverSpec{
            "square", 1,
            std::type_index(typeid(localization::AmbiguousEstimateMessage)),
            [](const auto& constants) {
              return std::make_unique<localization::SquareSolverNode>(
                  "detections", "estimate", constants[0].intrinsics_path,
                  constants[0].extrinsics_path);
            }},
        SolverSpec{
            "multi_tag", 1,
            std::type_index(typeid(localization::AmbiguousEstimateMessage)),
            [](const auto& constants) {
              return std::make_unique<localization::MultiTagSolverNode>(
                  "detections", "estimate", constants[0].intrinsics_path,
                  constants[0].extrinsics_path);
            }},
        SolverSpec{
            "unambiguous", 1,
            std::type_index(typeid(localization::PositionEstimate)),
            [](const auto& constants) {
              return std::make_unique<localization::UnambiguousSolverNode>(
                  "estimate", constants);
            }},
        SolverSpec{
            "joint", 1, std::type_index(typeid(localization::PositionEstimate)),
            [](const auto& constants) {
              return std::make_unique<localization::JointSolverNode>(
                  "estimate", constants);
            }}),
    [](const testing::TestParamInfo<SolverSpec>& info) {
      return info.param.name;
    });

}  // namespace
