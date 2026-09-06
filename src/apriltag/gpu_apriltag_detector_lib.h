#include <apriltag.h>
#include <array>
#include <cstddef>
#include <cstdint>
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
};

struct ApriltagDetection {
  Quad quad;
  int id;
};

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

auto GetSegments(ImageView32 segmented_apriltag)
    -> std::vector<std::vector<Coord<int>>>;

void PopulateBoundarySegmentedApriltag(
    std::vector<std::vector<Coord<int>>>& segments,
    ImageView32 boundary_segmented_apriltag);

auto SortSegments(std::vector<std::vector<Coord<int>>>& segments);

void PopulateSortedBoundarySegmentedApriltag(
    std::vector<std::vector<Coord<int>>>& segments,
    ImageView sorted_boundary_segmented_apriltag);

auto GetMses(std::vector<std::vector<Coord<int>>>& segments)
    -> std::vector<std::vector<float>>;

auto GetCandidatesQuadCorners(
    const std::vector<std::vector<Coord<int>>>& segments,
    const std::vector<std::vector<float>>& mse_map)
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

auto GetTagIds(std::vector<BitLocation>& bit_locations, ImageView apriltag,
               apriltag_family_t* family)
    -> std::pair<std::vector<int>, std::vector<int>>;

void RotateQuads(std::vector<Quad>& quads, std::vector<int>& rotations);

void DrawTagDetections(cv::Mat& image,
                       const std::vector<ApriltagDetection>& detections);

auto DetectAprilTag(ImageView apriltag, bool imwrite = true)
    -> std::vector<ApriltagDetection>;
}  // namespace apriltag
