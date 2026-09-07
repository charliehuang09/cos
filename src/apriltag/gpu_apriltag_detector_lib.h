#pragma once

#include <apriltag.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <opencv2/core/mat.hpp>
#include <string>
#include <utility>
#include <vector>
namespace apriltag {

template <typename T>
struct Coord {
  T row;
  T col;

  friend auto operator==(const Coord&, const Coord&) -> bool = default;

  template <typename H>
  friend auto AbslHashValue(H h, const Coord& coord) -> H {
    return H::combine(std::move(h), coord.row, coord.col);
  }
};

using BitLocation = std::array<std::array<Coord<int>, 10>, 10>;

struct ImageView {
  uint8_t* data;
  uint8_t* data_gpu = nullptr;
  int stride;
  int height;
  int width;
  void EnableGpu();
  auto operator()(size_t row, size_t col) -> uint8_t& {
    return data[row * stride + col];
  }
};

struct ImageView32 {
  uint32_t* data;
  int stride;
  int height;
  int width;

  auto operator()(size_t row, size_t col) -> uint32_t& {
    return data[row * stride + col];
  }
};

struct Quad {
  std::array<Coord<int>, 4> corners{};
};

struct CandidatesQuad {
  std::array<Coord<int>, 4> corners{};
}; struct ApriltagDetection { Quad quad; int id; };

struct WeightedPoint {
  Coord<int> coord;
  float weight;
};

void ImWrite(const std::string& path, const ImageView& image);

void ImWrite(const std::string& path, const ImageView& image_r,
             const ImageView& image_g, const ImageView& image_b);

void ImWrite(const std::string& path, ImageView32 segmented_apriltag);

void PopulateMinMax(ImageView apriltag, ImageView min, ImageView max);
void PopulateMinMaxGPU(ImageView apriltag, ImageView min, ImageView max);

void PopulateThresholdValid(ImageView min, ImageView max, ImageView threshold,
                            ImageView valid);
void PopulateThresholdValidGPU(ImageView min, ImageView max,
                               ImageView threshold, ImageView valid);

void PopulateBinarizedApriltag(ImageView threshold, ImageView valid,
                               ImageView apriltag,
                               ImageView binarized_apriltag);

void Segment(int row, int col, ImageView binarized_apriltag,
             ImageView32 segmented_apriltag, int32_t id);

void PopulateSegmentedApriltag(ImageView binarized_apriltag,
                               ImageView32 segmented_apriltag);

void PopulateSegmentedApriltagGPU(ImageView binarized_apriltag,
                                  ImageView32 segmented_apriltag,
                                  uint8_t* device_image,
                                  uint32_t* device_labels);

// Enqueue preprocessing and labeling on the default CUDA stream. Buffers are
// owned by the caller and must remain valid until the stream finishes.
void PopulatePreprocessedApriltagGPU(ImageView image, ImageView min, ImageView max,
                                    ImageView threshold, ImageView valid,
                                    uint8_t* device_binarized);
void PopulateSegmentedApriltagDevice(const uint8_t* device_image,
                                     uint32_t* device_labels, int width, int height);

auto GetSegments(ImageView32 segmented_apriltag)
    -> std::vector<std::vector<Coord<int>>>;

auto GetSegmentsCPU(ImageView32 segmented_apriltag)
    -> std::vector<std::vector<Coord<int>>>;

class GpuSegmentExtractor {
 public:
  GpuSegmentExtractor();
  ~GpuSegmentExtractor();
  GpuSegmentExtractor(const GpuSegmentExtractor&) = delete;
  auto operator=(const GpuSegmentExtractor&) -> GpuSegmentExtractor& = delete;
  GpuSegmentExtractor(GpuSegmentExtractor&&) noexcept;
  auto operator=(GpuSegmentExtractor&&) noexcept -> GpuSegmentExtractor&;

  auto Extract(ImageView32 labels) -> std::vector<std::vector<Coord<int>>>;
  // labels is device memory; stride is measured in uint32_t elements.
  auto ExtractDevice(const uint32_t* labels, int width, int height, int stride)
      -> std::vector<std::vector<Coord<int>>>;
  // Retained device allocations, excluding CUDA runtime overhead.
  auto DeviceWorkspaceBytes() const -> size_t;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void PopulateBoundarySegmentedApriltag(
    std::vector<std::vector<Coord<int>>>& segments,
    ImageView32 boundary_segmented_apriltag);

class GpuSegmentSorter {
 public:
  GpuSegmentSorter();
  ~GpuSegmentSorter();
  GpuSegmentSorter(const GpuSegmentSorter&) = delete;
  auto operator=(const GpuSegmentSorter&) -> GpuSegmentSorter& = delete;
  GpuSegmentSorter(GpuSegmentSorter&&) noexcept;
  auto operator=(GpuSegmentSorter&&) noexcept -> GpuSegmentSorter&;

  void Sort(std::vector<std::vector<Coord<int>>>& segments);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

void SortSegments(std::vector<std::vector<Coord<int>>>& segments);
void SortSegmentsGPU(std::vector<std::vector<Coord<int>>>& segments);
void SortSegmentsCPU(std::vector<std::vector<Coord<int>>>& segments);

void PopulateSortedBoundarySegmentedApriltag(
    std::vector<std::vector<Coord<int>>>& segments,
    ImageView sorted_boundary_segmented_apriltag);

auto GetMses(std::vector<std::vector<Coord<int>>>& segments)
    -> std::vector<std::vector<float>>;

auto GetCandidatesQuadCorners(
    const std::vector<std::vector<Coord<int>>>& segments,
    const std::vector<std::vector<float>>& mse_map)
    -> std::vector<CandidatesQuad>;

auto GetCandidatesQuadCornersParallel(
    const std::vector<std::vector<Coord<int>>>& segments)
    -> std::vector<CandidatesQuad>;

void PopulateCandidateQuadCornersApriltagBuffer(
    std::vector<CandidatesQuad>& quads,
    ImageView candidates_quad_corners_apriltag);

auto GetQuads(std::vector<CandidatesQuad>& candidate_quad_corners)
    -> std::vector<Quad>;

void PopulateQuadApriltagBuffer(std::vector<Quad>& quads,
                                ImageView quad_apriltag);

auto GetBitLocations(std::vector<Quad>& quads) -> std::vector<BitLocation>;

void PopulateBitLocationsApriltag(std::vector<BitLocation>& bit_locations,
                                  ImageView32 bit_locations_apriltag);

auto GetBlackWhiteThreshold(ImageView apriltag,
                            const BitLocation& bit_location);

class GpuTagIdDecoder {
 public:
  GpuTagIdDecoder();
  ~GpuTagIdDecoder();
  GpuTagIdDecoder(const GpuTagIdDecoder&) = delete;
  auto operator=(const GpuTagIdDecoder&) -> GpuTagIdDecoder& = delete;
  GpuTagIdDecoder(GpuTagIdDecoder&&) noexcept;
  auto operator=(GpuTagIdDecoder&&) noexcept -> GpuTagIdDecoder&;

  void SetTargetCodes(apriltag_family_t* family,
                      const std::vector<int>& target_tag_ids);
  auto Decode(const std::vector<BitLocation>& bit_locations,
              ImageView apriltag)
      -> std::pair<std::vector<int>, std::vector<int>>;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

auto GetTagIds(std::vector<BitLocation>& bit_locations, ImageView apriltag,
               apriltag_family_t* family,
               const std::vector<int>& target_tag_ids = {})
    -> std::pair<std::vector<int>, std::vector<int>>;

auto GetTagIdsGPU(const std::vector<BitLocation>& bit_locations,
                  ImageView apriltag,
                  apriltag_family_t* family,
                  const std::vector<int>& target_tag_ids = {})
    -> std::pair<std::vector<int>, std::vector<int>>;

auto GetTagIdsCPU(std::vector<BitLocation>& bit_locations, ImageView apriltag,
                  apriltag_family_t* family,
                  const std::vector<int>& target_tag_ids = {})
    -> std::pair<std::vector<int>, std::vector<int>>;

void RotateQuads(std::vector<Quad>& quads, std::vector<int>& rotations);

void DrawTagDetections(cv::Mat& image,
                       const std::vector<ApriltagDetection>& detections);

auto DetectAprilTag(ImageView apriltag, bool imwrite = true,
                    const std::vector<int>& target_tag_ids = {})
    -> std::vector<ApriltagDetection>;

auto GetRefinedPoints(const std::vector<ApriltagDetection>& apriltag_detections,
                      ImageView& apriltag)
    -> std::vector<std::array<std::vector<WeightedPoint>, 4>>;

auto GetRefinedQuads(
    const std::vector<std::array<std::vector<WeightedPoint>, 4>>&
        refined_points) -> std::vector<Quad>;

void PopulateRefinedPointsApriltag(
    const std::vector<std::array<std::vector<WeightedPoint>, 4>>&
        refined_points,
    ImageView& refined_points_apriltag);
}  // namespace apriltag
