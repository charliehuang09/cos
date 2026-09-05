cmake_minimum_required(VERSION 3.16) 
set(CMAKE_WARN_DEPRECATED OFF CACHE BOOL "" FORCE)

project(COS CXX C CUDA)

enable_testing()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)


set(CMAKE_BUILD_RPATH "$ORIGIN/../lib")
set(CMAKE_INSTALL_RPATH "$ORIGIN/../lib")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include(cmake/Clangd.cmake)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,GNU>:-Wno-psabi>)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 87)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-march=armv8-a+simd>)

list(PREPEND CMAKE_PREFIX_PATH "${CMAKE_SYSROOT}/opt/nvidia/vpi4")

find_package(CUDAToolkit REQUIRED)
find_package(JPEG REQUIRED)
find_package(OpenCV REQUIRED)
find_package(wpilib REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(vpi REQUIRED)
find_package(libuvc REQUIRED)

add_subdirectory(third_party)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
  DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  ixwebsocket
  GIT_REPOSITORY https://github.com/machinezone/IXWebSocket.git
  GIT_TAG v12.0.1
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(ixwebsocket)
target_compile_options(ixwebsocket PRIVATE
    -Wno-array-bounds
)

add_subdirectory(src)

include(cmake/DevOrin.cmake)
