#pragma once

#include <apriltag.h>
#include <cuda_runtime.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <memory_resource>
#include <opencv2/core/mat.hpp>
#include <optional>
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

template <typename T>
struct ImageView {
  T* data;
  // Stride is measured in elements, not bytes.
  int stride;
  int height;
  int width;

  auto operator()(size_t row, size_t col) -> T& {
    return data[row * stride + col];
  }
};

struct Quad {
  std::array<Coord<int16_t>, 4> corners{};
};

struct CandidatesQuad {
  std::array<Coord<int16_t>, 4> corners{};
};

struct ApriltagDetection {
  Quad quad;
  int id;
};

struct WeightedPoint {
  Coord<int> coord;
  float weight;
};

// Detects tag36h11 AprilTags and owns the tag family and working memory.
// Reuse one detector across frames; calls on the same instance must be serialized.
class GpuApriltagDetector {
 public:
  // Allocates image buffers for a fixed, positive size divisible by four.
  GpuApriltagDetector(int width, int height);
  ~GpuApriltagDetector();

  GpuApriltagDetector(const GpuApriltagDetector&) = delete;
  auto operator=(const GpuApriltagDetector&) -> GpuApriltagDetector& = delete;
  GpuApriltagDetector(GpuApriltagDetector&&) = delete;
  auto operator=(GpuApriltagDetector&&) -> GpuApriltagDetector& = delete;

  // Runs the entire pipeline and returns refined detections. The input is
  // borrowed only for this call and must match the constructor dimensions.
  // Image buffers are cleared and reused without resizing.
  // A nonempty output_directory enables
  // debug images; the directory is created if needed. An empty path disables them.
  auto DetectAprilTag(ImageView<uint8_t> apriltag,
                      const std::filesystem::path& output_directory = {})
      -> std::vector<ApriltagDetection>;

  void DrawTagDetections(cv::Mat& image,
                         const std::vector<ApriltagDetection>& detections);

 private:
  void SyncStream();

  void FreeBuffers();
  void ClearBuffers();

  void ImWrite(const std::string& path, const ImageView<uint8_t>& image);

  void ImWrite(const std::string& path, const ImageView<uint8_t>& image_r,
               const ImageView<uint8_t>& image_g,
               const ImageView<uint8_t>& image_b);

  void ImWrite(const std::string& path, ImageView<uint32_t> segmented_apriltag);

  void RegisterApriltagViewToGPU(ImageView<uint8_t> apriltag);
  void UnregisterApriltagViewToGPU(ImageView<uint8_t> apriltag);

  void PopulateMinMax(ImageView<uint8_t> apriltag, ImageView<uint8_t> min,
                      ImageView<uint8_t> max);
  void PopulateMinMaxGPU(ImageView<uint8_t> apriltag, ImageView<uint8_t> min,
                         ImageView<uint8_t> max, cudaStream_t stream);

  void PopulateThresholdValid(ImageView<uint8_t> min, ImageView<uint8_t> max,
                              ImageView<uint8_t> threshold,
                              ImageView<uint8_t> valid);
  void PopulateThresholdValidGPU(ImageView<uint8_t> apriltag,
                                 ImageView<uint8_t> min, ImageView<uint8_t> max,
                                 ImageView<uint8_t> binarized_apriltag,
                                 cudaStream_t stream);

  void PopulateBinarizedApriltag(ImageView<uint8_t> threshold,
                                 ImageView<uint8_t> valid,
                                 ImageView<uint8_t> apriltag,
                                 ImageView<uint8_t> binarized_apriltag);

  void Segment(int row, int col, ImageView<uint8_t> binarized_apriltag,
               ImageView<uint32_t> segmented_apriltag, int32_t id);

  void PopulateSegmentedApriltag(ImageView<uint8_t> binarized_apriltag,
                                 ImageView<uint32_t> segmented_apriltag);
  void PopulateSegmentedApriltagGPU(ImageView<uint8_t> binarized_apriltag,
                                    ImageView<uint32_t> segmented_apriltag,
                                    ImageView<uint32_t> dsu,
                                    cudaStream_t stream);

  auto GetSegment(ImageView<uint32_t>& segmented_apriltag, int16_t row,
                  int16_t col) -> std::pmr::vector<Coord<int16_t>>;
  auto GetSegments(ImageView<uint32_t> segmented_apriltag)
      -> std::pmr::vector<std::pmr::vector<Coord<int16_t>>>;

  void PopulateBoundarySegmentedApriltag(
      std::pmr::vector<std::pmr::vector<Coord<int16_t>>>& segments,
      ImageView<uint32_t> boundary_segmented_apriltag);

  auto SortSegments(std::pmr::vector<std::pmr::vector<Coord<int>>>& segments);

  void PopulateSortedBoundarySegmentedApriltag(
      std::pmr::vector<std::pmr::vector<Coord<int16_t>>>& segments,
      ImageView<uint8_t> sorted_boundary_segmented_apriltag);

  auto GetMses(std::pmr::vector<std::pmr::vector<Coord<int16_t>>>& segments)
      -> std::vector<std::vector<float>>;

  auto GetCandidatesQuadCorners(
      const std::pmr::vector<std::pmr::vector<Coord<int16_t>>>& segments,
      const std::vector<std::vector<float>>& mse_map)
      -> std::vector<CandidatesQuad>;

  void PopulateCandidateQuadCornersApriltagBuffer(
      std::vector<CandidatesQuad>& quads,
      ImageView<uint8_t> candidates_quad_corners_apriltag);

  auto GetQuads(std::vector<CandidatesQuad>& candidate_quad_corners)
      -> std::vector<Quad>;

  void OrderQuads(std::vector<Quad>& quads);

  void PopulateQuadApriltagBuffer(std::vector<Quad>& quads,
                                  ImageView<uint8_t> quad_apriltag);

  auto GetBitLocations(std::vector<Quad>& quads) -> std::vector<BitLocation>;
  auto GetBitLocationsHomography(std::vector<Quad>& quads, int width,
                                 int height) -> std::vector<BitLocation>;

  void PopulateBitLocationsApriltag(std::vector<BitLocation>& bit_locations,
                                    ImageView<uint32_t> bit_locations_apriltag);

  auto GetBlackWhiteThreshold(ImageView<uint8_t> apriltag,
                              const BitLocation& bit_location)
      -> std::array<float, 3>;

  auto GetTagIds(std::vector<BitLocation>& bit_locations,
                 ImageView<uint8_t> apriltag)
      -> std::pair<std::vector<int>, std::vector<int>>;

  void RotateQuads(std::vector<Quad>& quads, const std::vector<int>& rotations);

  auto GradientCol(Coord<int> point, ImageView<uint8_t>& apriltag) -> float;

  auto GradientRow(Coord<int> point, ImageView<uint8_t>& apriltag) -> float;

  auto GetRefinedPoints(
      const std::vector<ApriltagDetection>& apriltag_detections,
      ImageView<uint8_t>& apriltag)
      -> std::vector<std::array<std::vector<WeightedPoint>, 4>>;

  void PopulateRefinedPointsApriltag(
      const std::vector<std::array<std::vector<WeightedPoint>, 4>>&
          refined_points,
      ImageView<uint8_t>& refined_points_apriltag);

  auto Cross(const Coord<float>& a, const Coord<float>& b) -> float;

  auto GetIntersection(const Coord<float>& centroid_a,
                       const std::pair<float, float>& vector_a,
                       const Coord<float>& centroid_b,
                       const std::pair<float, float>& vector_b)
      -> Coord<int16_t>;

  auto GetRefinedQuads(
      const std::vector<std::array<std::vector<WeightedPoint>, 4>>&
          refined_points) -> std::vector<Quad>;
  void CreateCudaGraph();

  int width_;
  int height_;
  std::unique_ptr<apriltag_family_t, void (*)(apriltag_family_t*)> family_;

  cudaGraph_t graph_;
  cudaGraphExec_t graph_exec_;

  // Packed image buffers; their stride is independent of the input stride.
  uint8_t* graph_input_buffer_ = nullptr;
  uint8_t* max_buffer_ = nullptr;
  uint8_t* min_buffer_ = nullptr;
  uint8_t* threshold_buffer_ = nullptr;
  uint8_t* valid_buffer_ = nullptr;
  uint8_t* binarized_apriltag_buffer_ = nullptr;
  uint32_t* segmented_apriltag_buffer_ = nullptr;
  uint32_t* boundary_segmented_apriltag_buffer_ = nullptr;
  uint8_t* sorted_boundary_segmented_apriltag_buffer_ = nullptr;
  uint8_t* candidate_quad_corners_apriltag_buffer_ = nullptr;
  uint8_t* quad_apriltag_buffer_ = nullptr;
  uint32_t* bit_locations_apriltag_buffer_ = nullptr;
  uint8_t* refined_points_apriltag_buffer_ = nullptr;
  uint8_t* debug_r_buffer_ = nullptr;
  uint8_t* debug_g_buffer_ = nullptr;
  uint8_t* debug_b_buffer_ = nullptr;
  uint32_t* dsu_buffer_;

  // Reuse views into the owned image buffers.
  ImageView<uint8_t> segmented_apriltag_r_view_{};
  ImageView<uint8_t> segmented_apriltag_g_view_{};
  ImageView<uint8_t> segmented_apriltag_b_view_{};
  ImageView<uint8_t> max_view_{};
  ImageView<uint8_t> min_view_{};
  ImageView<uint8_t> binarized_apriltag_view_{};
  ImageView<uint8_t> threshold_view_{};
  ImageView<uint8_t> valid_view_{};
  ImageView<uint32_t> segmented_apriltag_view_{};
  ImageView<uint32_t> boundary_segmented_apriltag_view_{};
  ImageView<uint8_t> sorted_boundary_segmented_apriltag_view_{};
  ImageView<uint8_t> candidate_quad_corners_apriltag_view_{};
  ImageView<uint8_t> quad_apriltag_view_{};
  ImageView<uint32_t> bit_locations_apriltag_view_{};
  ImageView<uint8_t> refined_points_apriltag_view_{};
  ImageView<uint32_t> dsu_view_{};
  ImageView<uint8_t> graph_input_view_{};

  // Declared before segments_ so the pool outlives the segment allocations.
  // Calls on this detector are serialized, so the pool needs no locking.
  // The constructor sizes the block limit for one coordinate per image pixel,
  // with a floor covering the 1024-point reservation's 2048-point growth.
  std::pmr::unsynchronized_pool_resource segment_resource_;

  std::pmr::vector<std::pmr::vector<Coord<int16_t>>> segments_{
      &segment_resource_};
  // One byte per packed-image pixel ID, cleared for each GetSegments call.
  std::vector<uint8_t> visited_segment_ids_;
  std::vector<std::vector<float>> mses_;
  std::vector<CandidatesQuad> candidate_quad_corners_;
  std::vector<Quad> quads_;
  std::vector<BitLocation> bit_locations_;
  std::vector<ApriltagDetection> detections_;
  std::vector<std::array<std::vector<WeightedPoint>, 4>> refined_points_;
  std::vector<Quad> refined_quads_;
  cudaStream_t stream_;
};

}  // namespace apriltag
