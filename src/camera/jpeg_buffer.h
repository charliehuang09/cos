#pragma once

#include <cstddef>
#include <cstdlib>
#include <typeinfo>

#include "control_loop/message.h"

namespace camera {

class JpegBuffer final : public control_loop::IMessage {
 public:
  JpegBuffer() : size(0), timestamp(0), ptr(nullptr) {}
  JpegBuffer(size_t size, double timestamp)
      : size(size),
        timestamp(timestamp),
        ptr(static_cast<unsigned char*>(std::malloc(size))) {}
  ~JpegBuffer() override { std::free(ptr); }
  JpegBuffer(const JpegBuffer&) = delete;
  JpegBuffer(JpegBuffer&& other) noexcept
      : size(other.size), timestamp(other.timestamp), ptr(other.ptr) {
    other.ptr = nullptr;
  }

  size_t size;
  double timestamp;
  unsigned char* ptr;
  auto GetType() -> const std::type_info& override {
    return typeid(JpegBuffer);
  }
  auto GetSize() -> size_t override { return sizeof(*this) + size; }
};

}  // namespace camera
