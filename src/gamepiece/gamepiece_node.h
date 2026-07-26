#pragma once

#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <frc/geometry/Pose3d.h>
#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>

#include "camera/nvjpeg_decode_node.h"
#include "control_loop/node.h"
#include "control_loop/thread_pool.h"
#include "gamepiece/gamepiece_detection.h"
#include "gamepiece/object_detector.h"

namespace gamepiece {

class GamepieceNode final : public control_loop::INode {
 public:
  GamepieceNode(std::unique_ptr<ObjectDetector> detector,
                std::string_view input_path, std::string_view output_path,
                const nlohmann::json& intrinsics,
                const nlohmann::json& extrinsics,
                control_loop::ThreadPool& thread_pool);

  auto CreateCallback()
      -> std::function<void(const control_loop::Context&)> override;
  void RegisterCallback(
      const std::function<void(const control_loop::Context&)>& callback)
      override;
  [[nodiscard]] auto GetDependencies() const
      -> const std::vector<control_loop::MessageDescriptor>& override;
  [[nodiscard]] auto GetPublications() const
      -> const std::vector<control_loop::MessageDescriptor>& override;

 private:
  auto RunDetection(const camera::DecodedJpegBuffer& frame)
      -> GamepieceDetections;
  auto ComputePose(const cv::Rect& bbox) const -> frc::Pose3d;

  std::unique_ptr<ObjectDetector> detector_;
  std::string input_path_;
  std::string output_path_;
  control_loop::ThreadPool& thread_pool_;
  std::vector<std::function<void(const control_loop::Context&)>> callbacks_;
  std::vector<control_loop::MessageDescriptor> dependencies_;
  std::vector<control_loop::MessageDescriptor> publications_;
  std::mutex detection_mutex_;

  float cam_cx_ = 0.0F;
  float cam_cy_ = 0.0F;
  float fx_ = 0.0F;
  float fy_ = 0.0F;
  float cam_pitch_ = 0.0F;
  float pinhole_height_ = 0.0F;
  frc::Pose3d cam_pose_;
};

}  // namespace gamepiece
