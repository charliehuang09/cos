#pragma once
#include <filesystem>

#include "gpu_apriltag_detector_lib.h"

namespace apriltag {
class GPUApriltagDetector {
 public:
  GPUApriltagDetector(int width, int height);
  GPUApriltagDetector(const GPUApriltagDetector&) = delete;
  auto operator=(const GPUApriltagDetector&) -> GPUApriltagDetector& = delete;
  GPUApriltagDetector(GPUApriltagDetector&&) = delete;
  auto operator=(GPUApriltagDetector&&) -> GPUApriltagDetector& = delete;
  ~GPUApriltagDetector();
  auto Detect(ImageView apriltag_view, bool generate_debug_image = false)
      -> std::vector<ApriltagDetection>;
  void WriteLogImages(const std::filesystem::path& log_path) const;

 private:
  int width_;
  int height_;
  apriltag_family_t* family_;
  ImageView max_view_;
  ImageView min_view_;
  ImageView threshold_view_;
  ImageView valid_view_;
  ImageView binarized_apriltag_view_;
  ImageView32 segmented_apriltag_view_;
  ImageView32 boundary_segmented_apriltag_view_;
  ImageView sorted_boundary_segmented_apriltag_view_;
  ImageView candidate_quad_corners_apriltag_view_;
  ImageView quad_apriltag_view_;
  ImageView32 bit_locations_apriltag_view_;
  ImageView refined_points_apriltag_view_;
};
}  // namespace apriltag
