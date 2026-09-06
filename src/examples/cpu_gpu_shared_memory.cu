#include <iostream>
#include <ostream>
#define CUDA_CHECK(call)                                                   \
    do {                                                                   \
        cudaError_t error = (call);                                        \
        if (error != cudaSuccess) {                                        \
            std::cerr << cudaGetErrorString(error) << '\n';                 \
            std::exit(EXIT_FAILURE);                                       \
        }                                                                  \
    } while (0)
int main() {
  float* a;
  CUDA_CHECK(cudaHostAlloc<float>(&a, sizeof(float), cudaHostAllocMapped));
  float *b;
  CUDA_CHECK(cudaHostGetDevicePointer(&b, a, 0));
  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << cudaFreeHost(a);
}