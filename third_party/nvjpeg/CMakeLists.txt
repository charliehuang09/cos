set(SAMPLES_ROOT /usr/src/jetson_multimedia_api)
set(COMMON_DIR ${SAMPLES_ROOT}/samples/common)

file(GLOB CLASS_SOURCES
    ${COMMON_DIR}/classes/*.cpp
)

add_library(nvjpeg
    ${CLASS_SOURCES}
)

target_include_directories(nvjpeg PUBLIC 
    ${SAMPLES_ROOT}/include
    ${SAMPLES_ROOT}/include/libjpeg-9f
    ${COMMON_DIR}/classes
    ${COMMON_DIR}/algorithm/cuda
    ${COMMON_DIR}/algorithm/trt
    /usr/include/aarch64-linux-gnu
    /usr/include/libdrm
)

set(NVIDIA_LIBDIR /usr/lib/aarch64-linux-gnu/nvidia)

set(CUDA_TOOLKIT_ROOT /usr/local/cuda-13.2)
find_path(CUDA_INCLUDE_DIR cuda.h
    HINTS
        ${CUDA_TOOLKIT_ROOT}/targets/aarch64-linux/include
        ${CUDA_TOOLKIT_ROOT}/targets/sbsa-linux/include
        ${CUDA_TOOLKIT_ROOT}/include
)

target_include_directories(nvjpeg PRIVATE ${CUDA_INCLUDE_DIR})

find_library(NVBUF_SURFACE_LIB nvbufsurface
    HINTS ${NVIDIA_LIBDIR}
)
find_library(NVBUF_SURFTRANSFORM_LIB nvbufsurftransform
    HINTS ${NVIDIA_LIBDIR}
)
find_library(NVMM_JPEG_LIB nvmm_jpeg
    HINTS ${NVIDIA_LIBDIR}
)
find_library(NVV4L2_LIB nvv4l2
    HINTS ${NVIDIA_LIBDIR}
)
find_library(NVOSD_LIB nvosd
    HINTS ${NVIDIA_LIBDIR}
)

find_library(CUDA_LIB cuda
    HINTS
        ${CUDA_TOOLKIT_ROOT}/targets/aarch64-linux/lib
        ${CUDA_TOOLKIT_ROOT}/targets/aarch64-linux/lib/stubs
        ${CUDA_TOOLKIT_ROOT}/targets/sbsa-linux/lib
        ${CUDA_TOOLKIT_ROOT}/targets/sbsa-linux/lib/stubs
)

target_link_libraries(nvjpeg PRIVATE
    pthread
    ${NVV4L2_LIB}
    EGL
    GLESv2
    X11
    ${NVBUF_SURFACE_LIB}
    ${NVBUF_SURFTRANSFORM_LIB}
    ${NVMM_JPEG_LIB}
    drm
    ${CUDA_LIB}
    vulkan
)
