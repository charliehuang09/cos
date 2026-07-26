#pragma once

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <string>
#include <vector>

#include "gamepiece/object_detector.h"

namespace gamepiece {

class Yolo final : public ObjectDetector, private nvinfer1::ILogger {
 public:
  Yolo(const std::string& model_path,
       const std::vector<std::string>& class_names);
  ~Yolo() override;

  Yolo(const Yolo&) = delete;
  auto operator=(const Yolo&) -> Yolo& = delete;

  auto Detect(const cv::cuda::GpuMat& image)
      -> std::vector<LabeledBoundingBox> override;

 private:
  void log(Severity severity, const char* message) noexcept override;
  void Preprocess(const cv::cuda::GpuMat& image);
  auto Run(const cv::cuda::GpuMat& image) -> std::vector<float>;
  auto Postprocess(int original_height, int original_width,
                   const std::vector<float>& results) const
      -> std::vector<LabeledBoundingBox>;

  const std::vector<std::string> class_names_;
  nvinfer1::IRuntime* runtime_ = nullptr;
  nvinfer1::ICudaEngine* engine_ = nullptr;
  nvinfer1::IExecutionContext* context_ = nullptr;
  cudaStream_t inference_cuda_stream_ = nullptr;
  float* input_buffer_ = nullptr;
  float* output_buffer_ = nullptr;
  size_t output_size_ = 0;
};

}  // namespace gamepiece
