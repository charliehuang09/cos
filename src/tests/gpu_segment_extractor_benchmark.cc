#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <numeric>
#include <sstream>
#include <vector>

#include <cuda_runtime.h>
#include <opencv2/imgcodecs.hpp>

#include "absl/log/check.h"
#include "apriltag/gpu_apriltag_detector.h"

namespace {
using Clock = std::chrono::steady_clock;
using Segments = std::vector<std::vector<apriltag::Coord<int>>>;

void Canonicalize(Segments& segments) {
  std::sort(segments.begin(), segments.end(), [](const auto& a, const auto& b) {
    return std::lexicographical_compare(a.begin(), a.end(), b.begin(), b.end(),
        [](const auto& x, const auto& y) {
          return x.row != y.row ? x.row < y.row : x.col < y.col;
        });
  });
}

template <typename Operation>
double Time(Operation operation, int iterations) {
  for (int i = 0; i < 3; ++i) operation();
  const auto start = Clock::now();
  for (int i = 0; i < iterations; ++i) operation();
  return std::chrono::duration<double, std::milli>(Clock::now() - start).count()
      / iterations;
}

void Benchmark(const std::string& name, apriltag::ImageView32 labels) {
  uint32_t* device = nullptr;
  const size_t pitch = size_t(labels.width) * sizeof(uint32_t);
  CHECK_EQ(cudaMalloc(&device, pitch * labels.height), cudaSuccess);
  CHECK_EQ(cudaMemcpy2D(device, pitch, labels.data,
      size_t(labels.stride) * sizeof(uint32_t), pitch, labels.height,
      cudaMemcpyHostToDevice), cudaSuccess);
  apriltag::GpuSegmentExtractor extractor;
  auto gpu = extractor.ExtractDevice(device, labels.width, labels.height, labels.width);
  const size_t workspace_bytes = extractor.DeviceWorkspaceBytes();
  auto cpu = apriltag::GetSegmentsCPU(labels);
  Canonicalize(cpu);
  Canonicalize(gpu);
  CHECK(cpu == gpu) << name;
  const double cpu_ms = Time([&] { apriltag::GetSegmentsCPU(labels); }, 20);
  const double gpu_ms = Time([&] {
    extractor.ExtractDevice(device, labels.width, labels.height, labels.width);
  }, 20);
  const double upload_ms = Time([&] { extractor.Extract(labels); }, 20);
  std::cout << name << ",cpu_ms=" << cpu_ms << ",device_gpu_ms=" << gpu_ms
            << ",host_gpu_ms=" << upload_ms << ",segments=" << gpu.size()
            << ",device_workspace_bytes=" << workspace_bytes
            << '\n';
  CHECK_EQ(cudaFree(device), cudaSuccess);
}
}  // namespace

int main(int argc, char** argv) {
  constexpr int width = 1280, height = 800;
  std::vector<uint32_t> labels(width * height);
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      labels[row * width + col] = 1 + col / 160 + 8 * (row / 160);
    }
  }
  Benchmark("sparse", {labels.data(), width, height, width});
  for (int row = 0; row < height; ++row) {
    for (int col = 0; col < width; ++col) {
      labels[row * width + col] = 1 + (row + col) % 2;
    }
  }
  Benchmark("dense", {labels.data(), width, height, width});
  if (argc < 2) return 0;
  std::vector<std::filesystem::path> paths;
  if (std::filesystem::is_directory(argv[1])) {
    for (const auto& entry : std::filesystem::directory_iterator(argv[1])) {
      auto ext = entry.path().extension().string();
      if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") paths.push_back(entry.path());
    }
    std::sort(paths.begin(), paths.end());
  } else {
    paths.emplace_back(argv[1]);
  }
  CHECK(!paths.empty());
  std::vector<int> ids(32);
  std::iota(ids.begin(), ids.end(), 1);
  for (size_t i = 0; i < std::min<size_t>(paths.size(), 20); ++i) {
    auto image = cv::imread(paths[i].string(), cv::IMREAD_GRAYSCALE);
    CHECK(!image.empty());
    apriltag::GPUApriltagDetector detector(image.cols, image.rows, ids);
    apriltag::ImageView view{.data = image.data, .stride = static_cast<int>(image.step),
                             .height = image.rows, .width = image.cols};
    const double detect_ms = Time([&] { detector.Detect(view); }, 10);
    std::cout << paths[i].filename() << ",detect_ms=" << detect_ms << '\n';
    std::vector<std::string> detections;
    for (const auto& detection : detector.Detect(view)) {
      std::ostringstream text;
      text << detection.id;
      for (const auto& corner : detection.quad.corners) {
        text << ':' << corner.row << ':' << corner.col;
      }
      detections.push_back(text.str());
    }
    std::sort(detections.begin(), detections.end());
    std::cout << paths[i].filename() << ",detections=";
    for (const auto& detection : detections) std::cout << detection << ';';
    std::cout << '\n';
    Benchmark(paths[i].filename().string(), detector.GetSegmentedApriltagView());
  }
}
