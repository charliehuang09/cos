#pragma once

#include <cuda_runtime.h>

namespace utils {

auto CheckCuda(cudaError_t status) -> void;

}  // namespace utils
