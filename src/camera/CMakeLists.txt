add_library(uvc_camera_node)
target_sources(uvc_camera_node
  PRIVATE
    uvc_camera_node.cc
)
target_link_libraries(uvc_camera_node PRIVATE
  uvc
  absl::log
  nlohmann_json::nlohmann_json
  wpilibc
)

set(TEGRA_MMAPI_ROOT
  "${CMAKE_SYSROOT}/usr/src/debug/tegra-mmapi-samples/39.2.0-20260601141651"
  CACHE PATH "Path to the Jetson Multimedia API samples")
set(TEGRA_NVJPEG_HEADERS
  "${CMAKE_SYSROOT}/usr/src/debug/gstreamer1.0-plugins-nvjpeg/1.14.0-r39.2.0/nv_headers"
  CACHE PATH "Path to NVIDIA's accelerated libjpeg headers")

set(TEGRA_LIBJPEG_SHIM "${CMAKE_CURRENT_BINARY_DIR}/nvjpeg-headers/libjpeg-9f")
file(MAKE_DIRECTORY "${TEGRA_LIBJPEG_SHIM}")
foreach(header jpeglib.h jmorecfg.h jerror.h)
  configure_file(
    "${TEGRA_NVJPEG_HEADERS}/${header}"
    "${TEGRA_LIBJPEG_SHIM}/${header}"
    COPYONLY
  )
endforeach()
configure_file(
  "${PROJECT_SOURCE_DIR}/third_party/tegra-mmapi/libjpeg-9f/jconfig.h"
  "${TEGRA_LIBJPEG_SHIM}/jconfig.h"
  COPYONLY
)

find_package(Threads REQUIRED)
find_library(NVMM_JPEG_LIBRARY NAMES nvmm_jpeg REQUIRED)
find_library(NVBUF_SURFACE_LIBRARY NAMES nvbufsurface REQUIRED)
find_library(NVBUF_TRANSFORM_LIBRARY NAMES nvbufsurftransform REQUIRED)

add_library(jetson_mmjpeg STATIC)
target_sources(jetson_mmjpeg PRIVATE
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvJpegDecoder.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvElement.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvElementProfiler.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvBufSurface.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvBuffer.cpp"
  "${TEGRA_MMAPI_ROOT}/samples/common/classes/NvLogging.cpp"
)
target_include_directories(jetson_mmjpeg
  PUBLIC
    "${TEGRA_MMAPI_ROOT}/include"
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/nvjpeg-headers"
)
target_compile_definitions(jetson_mmjpeg PRIVATE TEGRA_ACCELERATE)
target_compile_options(jetson_mmjpeg PRIVATE -Wno-unused-parameter)
target_link_libraries(jetson_mmjpeg
  PUBLIC
    ${NVMM_JPEG_LIBRARY}
    ${NVBUF_SURFACE_LIBRARY}
    ${NVBUF_TRANSFORM_LIBRARY}
    CUDA::cuda_driver
    Threads::Threads
)

add_library(nvjpeg_decode_node)
target_sources(nvjpeg_decode_node
  PRIVATE
    nvjpeg_gpu_decode_node.cc
)
target_compile_options(nvjpeg_decode_node PRIVATE
  $<$<CXX_COMPILER_ID:GNU>:-Wno-array-bounds>
)
target_link_libraries(nvjpeg_decode_node
  PUBLIC
    CUDA::cudart
    CUDA::nvjpeg
    absl::log
  PRIVATE
    timer
)

add_library(nvjpeg_fd_decode_node)
target_sources(nvjpeg_fd_decode_node
  PRIVATE
    nvjpeg_hardware_accelerated_decode_node.cc
)
target_include_directories(nvjpeg_fd_decode_node PRIVATE
  "${CMAKE_CURRENT_BINARY_DIR}/nvjpeg-headers"
)
target_link_libraries(nvjpeg_fd_decode_node
  PUBLIC
    jetson_mmjpeg
    absl::log
  PRIVATE
    timer
)

add_library(jpeg_disk_camera)
target_sources(jpeg_disk_camera
  PRIVATE
    jpeg_disk_camera.cc
)
target_link_libraries(jpeg_disk_camera PRIVATE
  uvc
  absl::log
  nlohmann_json::nlohmann_json
  wpilibc
)
