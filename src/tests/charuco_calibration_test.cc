#include "tools/charuco_calibration.h"

#include <gtest/gtest.h>
#include <opencv2/core.hpp>

namespace {

using namespace charuco_calibration;

TEST(CharucoCalibrationTest, DetectedBoardSurvivesJsonRoundTrip) {
  const auto board = CreateBoard();
  const auto detected =
      DetectCharucoBoard(GenerateBoardImage(board), CreateDetector(board));
  ASSERT_TRUE(HasEnoughCorners(detected));
  // Include the text encoding/decoding step used by the two tools.
  const auto restored = DetectionFromJson(
      nlohmann::json::parse(DetectionToJson(detected).dump()));
  EXPECT_EQ(restored.charuco_corners.type(), CV_32FC2);
  EXPECT_EQ(restored.charuco_ids.type(), CV_32SC1);
  EXPECT_EQ(restored.charuco_corners.size(), detected.charuco_corners.size());
  EXPECT_EQ(restored.charuco_ids.size(), detected.charuco_ids.size());
  EXPECT_EQ(cv::norm(restored.charuco_corners, detected.charuco_corners), 0);
  EXPECT_EQ(cv::norm(restored.charuco_ids, detected.charuco_ids), 0);
  EXPECT_EQ(restored.image_points, detected.image_points);
  EXPECT_EQ(restored.object_points, detected.object_points);
}

TEST(CharucoCalibrationTest, FrameWithoutBoardIsPreserved) {
  const cv::Mat blank(480, 640, CV_8UC3, cv::Scalar(255, 255, 255));
  const auto detected = DetectCharucoBoard(blank, CreateDetector(CreateBoard()));
  const auto saved = DetectionToJson(detected);
  EXPECT_TRUE(saved.at("charuco_corners").empty());
  EXPECT_TRUE(saved.at("charuco_ids").empty());
  EXPECT_TRUE(saved.at("image_points").empty());
  EXPECT_TRUE(saved.at("object_points").empty());
  EXPECT_FALSE(HasEnoughCorners(DetectionFromJson(saved)));
}

TEST(CharucoCalibrationTest, PartialDetectionIsPreserved) {
  DetectionResult partial;
  partial.charuco_corners.push_back(cv::Point2f(12.5F, 24.0F));
  partial.charuco_ids.push_back(7);
  const auto restored = DetectionFromJson(DetectionToJson(partial));
  EXPECT_EQ(restored.charuco_corners.total(), 1U);
  EXPECT_EQ(restored.charuco_ids.at<int>(0), 7);
  EXPECT_FALSE(HasEnoughCorners(restored));
}

TEST(CharucoCalibrationTest, RejectsMismatchedPointArrays) {
  auto saved = DetectionToJson(DetectionResult{});
  saved["charuco_ids"].push_back(1);
  EXPECT_THROW(DetectionFromJson(saved), std::runtime_error);
  saved = DetectionToJson(DetectionResult{});
  saved["image_points"].push_back({1.0, 2.0});
  EXPECT_THROW(DetectionFromJson(saved), std::runtime_error);
}

TEST(CharucoCalibrationTest, RejectsCollinearBoardCorners) {
  const auto board = CreateBoard();
  for (const auto& ids : {std::vector<int>{0, 1, 2, 3},
                         std::vector<int>{0, 12, 24, 36}}) {
    DetectionResult result;
    for (int id : ids) {
      result.charuco_ids.push_back(id);
      result.charuco_corners.push_back(cv::Point2f(id * 10.0F, 100.0F));
    }
    board.matchImagePoints(result.charuco_corners, result.charuco_ids,
                          result.object_points, result.image_points);
    ASSERT_EQ(result.charuco_corners.total(), 4U);
    EXPECT_FALSE(HasEnoughCorners(result));
    EXPECT_FALSE(HasEnoughCorners(DetectionFromJson(DetectionToJson(result))));
  }
}

}  // namespace
