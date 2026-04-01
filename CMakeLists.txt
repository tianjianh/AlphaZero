cmake_minimum_required(VERSION 3.16)
project(MiniGo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ── Build type ──────────────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── Platform-aware optimisation flags ───────────────────────
if(CMAKE_BUILD_TYPE STREQUAL "Release")
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -mcpu=native -DNDEBUG")
  else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -DNDEBUG")
  endif()
endif()

# ── Find Eigen3 ────────────────────────────────────────────
list(APPEND CMAKE_PREFIX_PATH
  /opt/homebrew
  /usr/local
)
find_package(Eigen3 REQUIRED NO_MODULE)

# ── Thread support ──────────────────────────────────────────
find_package(Threads REQUIRED)

# ── Apple Accelerate (free BLAS — speeds up Eigen GEMM) ────
if(APPLE)
  find_library(ACCELERATE_LIB Accelerate)
  if(ACCELERATE_LIB)
    message(STATUS "Using Apple Accelerate for BLAS")
  endif()
endif()

# ── Backend selection (pick ONE) ────────────────────────────
#
# MINIGO_BACKEND controls which GPU backend is compiled:
#   auto   — auto-detect best available (default)
#   metal  — Metal/MPSGraph (macOS Apple Silicon only)
#   opencl — OpenCL 1.2+ (Linux NVIDIA/AMD/Intel, macOS)
#   eigen  — CPU only (no GPU)
#
# Usage:
#   cmake .. -DMINIGO_BACKEND=auto     # (default)
#   cmake .. -DMINIGO_BACKEND=metal
#   cmake .. -DMINIGO_BACKEND=opencl
#   cmake .. -DMINIGO_BACKEND=eigen
#
set(MINIGO_BACKEND "auto" CACHE STRING "Inference backend: auto, metal, opencl, eigen")
set_property(CACHE MINIGO_BACKEND PROPERTY STRINGS "auto" "metal" "opencl" "eigen")

# ── Auto-detect best backend ────────────────────────────────
if(MINIGO_BACKEND STREQUAL "auto")
  if(APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "arm64")
    # macOS Apple Silicon: try Metal first
    find_library(METAL_FRAMEWORK Metal)
    find_library(FOUNDATION_FRAMEWORK Foundation)
    find_library(MPS_FRAMEWORK MetalPerformanceShaders)
    find_library(MPSG_FRAMEWORK MetalPerformanceShadersGraph)
    if(METAL_FRAMEWORK AND FOUNDATION_FRAMEWORK AND MPS_FRAMEWORK AND MPSG_FRAMEWORK)
      set(MINIGO_BACKEND "metal")
    else()
      find_package(OpenCL QUIET)
      if(OpenCL_FOUND)
        set(MINIGO_BACKEND "opencl")
      else()
        set(MINIGO_BACKEND "eigen")
      endif()
    endif()
  elseif(APPLE)
    # macOS Intel: try OpenCL
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
      set(MINIGO_BACKEND "opencl")
    else()
      set(MINIGO_BACKEND "eigen")
    endif()
  else()
    # Linux: try OpenCL
    find_package(OpenCL QUIET)
    if(OpenCL_FOUND)
      set(MINIGO_BACKEND "opencl")
    else()
      set(MINIGO_BACKEND "eigen")
    endif()
  endif()
endif()

# ── Validate and configure selected backend ─────────────────
set(HAS_OPENCL FALSE)
set(HAS_METAL FALSE)

if(MINIGO_BACKEND STREQUAL "metal")
  if(NOT APPLE)
    message(FATAL_ERROR "Metal backend requires macOS")
  endif()
  find_library(METAL_FRAMEWORK Metal)
  find_library(FOUNDATION_FRAMEWORK Foundation)
  find_library(MPS_FRAMEWORK MetalPerformanceShaders)
  find_library(MPSG_FRAMEWORK MetalPerformanceShadersGraph)
  if(NOT METAL_FRAMEWORK OR NOT FOUNDATION_FRAMEWORK OR NOT MPS_FRAMEWORK OR NOT MPSG_FRAMEWORK)
    message(FATAL_ERROR "Metal frameworks not found. Install Xcode Command Line Tools.")
  endif()
  set(HAS_METAL TRUE)

elseif(MINIGO_BACKEND STREQUAL "opencl")
  find_package(OpenCL REQUIRED)
  set(HAS_OPENCL TRUE)

elseif(MINIGO_BACKEND STREQUAL "eigen")
  # Nothing extra needed — Eigen is always available

else()
  message(FATAL_ERROR "Unknown backend: ${MINIGO_BACKEND}. Use: auto, metal, opencl, eigen")
endif()

# ── Library: minigo core ────────────────────────────────────
set(CORE_SOURCES
    src/game.cpp
    src/onnx_loader.cpp
    src/loaded_model.cpp
    src/compute_context.cpp
    src/eigen_compute.cpp
    src/nn_evaluator.cpp
    src/mcts.cpp
)

if(HAS_OPENCL)
  list(APPEND CORE_SOURCES src/opencl_compute.cpp)
endif()

if(HAS_METAL)
  list(APPEND CORE_SOURCES src/metal_compute.mm)
endif()

add_library(minigo_core STATIC ${CORE_SOURCES})
target_include_directories(minigo_core PUBLIC include)
target_link_libraries(minigo_core PUBLIC Eigen3::Eigen Threads::Threads)

if(APPLE AND ACCELERATE_LIB)
  target_compile_definitions(minigo_core PUBLIC EIGEN_USE_BLAS)
  target_link_libraries(minigo_core PUBLIC ${ACCELERATE_LIB})
endif()

if(HAS_OPENCL)
  target_compile_definitions(minigo_core PUBLIC MINIGO_HAS_OPENCL=1)
  target_link_libraries(minigo_core PUBLIC OpenCL::OpenCL)
endif()

if(HAS_METAL)
  target_compile_definitions(minigo_core PUBLIC MINIGO_HAS_METAL=1)
  target_link_libraries(minigo_core PUBLIC
      ${METAL_FRAMEWORK} ${FOUNDATION_FRAMEWORK}
      ${MPS_FRAMEWORK} ${MPSG_FRAMEWORK})
endif()

# ── Executables ─────────────────────────────────────────────
add_executable(play      src/main_play.cpp)
add_executable(selfplay  src/main_selfplay.cpp)
add_executable(benchmark src/main_benchmark.cpp)

target_link_libraries(play      PRIVATE minigo_core)
target_link_libraries(selfplay  PRIVATE minigo_core)
target_link_libraries(benchmark PRIVATE minigo_core)

# ── Summary ─────────────────────────────────────────────────
message(STATUS "System:     ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Backend:    ${MINIGO_BACKEND}")
message(STATUS "Eigen3:     ${EIGEN3_INCLUDE_DIR}")
