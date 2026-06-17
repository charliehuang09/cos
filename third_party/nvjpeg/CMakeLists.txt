add_library(nvjpeg)

target_sources(nvjpeg
  PRIVATE
    NvApplicationProfiler.cpp
    NvBufSurface.cpp
    NvBuffer.cpp
    NvDrmRenderer.cpp
    NvEglRenderer.cpp
    NvElement.cpp
    NvElementProfiler.cpp
    NvJpegDecoder.cpp
    NvJpegEncoder.cpp
    NvLogging.cpp
    NvUtils.cpp
    NvV4l2Element.cpp
    NvV4l2ElementPlane.cpp
    NvVideoDecoder.cpp
    NvVideoEncoder.cpp
)

target_include_directories(nvjpeg
  PUBLIC
    ${NVJPEG_SOURCE_DIR}
    ${JETSON_MULTIMEDIA_API_ROOT}/include
    ${CMAKE_SYSROOT}/usr/include/libdrm
    .
)

target_link_libraries(nvjpeg
  PUBLIC
    JPEG::JPEG
  PRIVATE
)
