#include "utils/cuda_utils.h"

#include "absl/log/check.h"

namespace utils {

auto CheckCuda(cudaError_t status) -> void {
  CHECK(status == cudaSuccess) << cudaGetErrorString(status);
}

}  // namespace utils
