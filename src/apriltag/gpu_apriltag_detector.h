#pragma once
#include <filesystem>

#include "gpu_apriltag_detector_lib.h"

namespace apriltag {
class GPUApriltagDetector {
 public:
  struct Profile {
    // Wall time per Detect call. Extraction includes input staging,
    // preprocessing, and labeling; decoding includes decimation retries.
    // Debug-image work, if requested, is charged to its containing stage.
    double extract_ms = 0;
    double sort_ms = 0;
    double quads_ms = 0;
    double decode_ms = 0;
    double refine_ms = 0;
  };
  GPUApriltagDetector(int width, int height,
                      std::vector<int> target_tag_ids = {},
                      int decimate = 1);
  GPUApriltagDetector(const GPUApriltagDetector&) = delete;
  auto operator=(const GPUApriltagDetector&) -> GPUApriltagDetector& = delete;
  GPUApriltagDetector(GPUApriltagDetector&&) = delete;
  auto operator=(GPUApriltagDetector&&) -> GPUApriltagDetector& = delete;
  ~GPUApriltagDetector();
  auto Detect(ImageView apriltag_view, bool generate_debug_image = false,
              Profile* profile = nullptr)
      -> std::vector<ApriltagDetection>;
  void WriteLogImages(const std::filesystem::path& log_path) const;
  // Copies the latest device labels to detector-owned host memory on demand.
  auto GetSegmentedApriltagView() const -> ImageView32;
  void SetTargetTagIds(std::vector<int> target_tag_ids) {
    target_tag_ids_ = std::move(target_tag_ids);
    tag_decoder_.SetTargetCodes(family_, target_tag_ids_);
  }
  auto GetTargetTagIds() const -> const std::vector<int>& {
    return target_tag_ids_;
  }
  auto GetDecimate() const -> int { return decimate_; }

 private:
  int width_;
  int height_;
  int full_width_;
  int full_height_;
  int decimate_ = 1;
  std::vector<int> target_tag_ids_;
  apriltag_family_t* family_;
  ImageView input_view_;
  bool has_frame_ = false;
  mutable bool segmented_host_dirty_ = false;
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
  uint8_t* device_image_ = nullptr;
  uint8_t* device_reduced_image_ = nullptr;
  uint32_t* device_labels_ = nullptr;
  GpuSegmentExtractor segment_extractor_;
  GpuSegmentSorter segment_sorter_;
  GpuTagIdDecoder tag_decoder_;
};
}  // namespace apriltag
