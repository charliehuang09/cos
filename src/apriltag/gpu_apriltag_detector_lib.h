#include <apriltag.h>
#include <array>
#include <cstddef>
#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>
namespace apriltag {

using BitLocation =
    std::array<std::array<std::pair<unsigned int, unsigned int>, 10>, 10>;

struct ImageView {
  uint8_t* data;
  unsigned int stride;
  unsigned int height;
  unsigned int width;

  auto operator()(size_t row, size_t col) -> uint8_t& {
    return data[row * stride + col];
  }
};

struct ImageView32 {
  uint32_t* data;
  unsigned int stride;
  unsigned int height;
  unsigned int width;

  auto operator()(size_t row, size_t col) -> uint32_t& {
    return data[row * stride + col];
  }
};

struct Quad {
  std::array<std::pair<int, int>, 4> corners{};
};

struct CandidatesQuad {
  std::array<std::pair<unsigned int, unsigned int>, 4> corners{};
};

struct ApriltagDetection {
  Quad quad;
  int id;
};

void ImWrite(const std::string& path, const ImageView& image);

void ImWrite(const std::string& path, const ImageView& image_r,
             const ImageView& image_g, const ImageView& image_b);

void ImWrite(const std::string& path, ImageView32 segmented_apriltag);

void PopulateMinMax(ImageView apriltag, ImageView min, ImageView max);

void PopulateThresholdValid(ImageView min, ImageView max, ImageView threshold,
                            ImageView valid);

void PopulateBinarizedApriltag(ImageView threshold, ImageView valid,
                               ImageView apriltag,
                               ImageView binarized_apriltag);

void Segment(uint row, uint col, ImageView binarized_apriltag,
             ImageView32 segmented_apriltag, int32_t id);

void PopulateSegmentedApriltag(ImageView binarized_apriltag,
                               ImageView32 segmented_apriltag);

auto GetSegments(ImageView32 segmented_apriltag)
    -> std::vector<std::vector<std::pair<uint, uint>>>;

void PopulateBoundarySegmentedApriltag(
    std::vector<std::vector<std::pair<uint, uint>>>& segments,
    ImageView32 boundary_segmented_apriltag);

auto SortSegments(std::vector<std::vector<std::pair<uint, uint>>>& segments);

void PopulateSortedBoundarySegmentedApriltag(
    std::vector<std::vector<std::pair<uint, uint>>>& segments,
    ImageView sorted_boundary_segmented_apriltag);

auto GetMses(std::vector<std::vector<std::pair<uint, uint>>>& segments)
    -> std::vector<std::vector<float>>;

auto GetCandidatesQuadCorners(
    const std::vector<std::vector<std::pair<uint, uint>>>& segments,
    const std::vector<std::vector<float>>& mse_map)
    -> std::vector<CandidatesQuad>;

void PopulateCandidateQuadCornersApriltagBuffer(
    std::vector<CandidatesQuad>& quads,
    ImageView candidates_quad_corners_apriltag);

auto GetQuads(std::vector<CandidatesQuad>& candidate_quad_corners)
    -> std::vector<Quad>;

void OrderQuads(std::vector<Quad>& quads);

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

auto DetectAprilTag(ImageView apriltag) -> std::vector<ApriltagDetection>;
}  // namespace apriltag
