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
#   cuda   — CUDA (NVIDIA GPUs)
#   opencl — OpenCL 1.2+ (Linux NVIDIA/AMD/Intel, macOS)
#   eigen  — CPU only (no GPU)
#
# Usage:
#   cmake .. -DMINIGO_BACKEND=auto     # (default)
#   cmake .. -DMINIGO_BACKEND=metal
#   cmake .. -DMINIGO_BACKEND=tensorrt
#   cmake .. -DMINIGO_BACKEND=cuda
#   cmake .. -DMINIGO_BACKEND=opencl
#   cmake .. -DMINIGO_BACKEND=eigen
#
set(MINIGO_BACKEND "auto" CACHE STRING "Inference backend: auto, metal, tensorrt, cuda, opencl, eigen")
set_property(CACHE MINIGO_BACKEND PROPERTY STRINGS "auto" "metal" "tensorrt" "cuda" "opencl" "eigen")

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
    # Linux: try TensorRT > CUDA > OpenCL
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
      # Detect GPU SM version to decide backend.
      # TensorRT 10.x requires SM >= 7.5 (Turing+).
      # V100 (SM 7.0) and older must use the CUDA backend.
      set(_trt_sm_ok FALSE)
      find_program(NVIDIA_SMI nvidia-smi)
      if(NVIDIA_SMI)
        execute_process(
          COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader,nounits
          OUTPUT_VARIABLE _gpu_caps
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET
          RESULT_VARIABLE _smi_rc)
        if(_smi_rc EQUAL 0 AND _gpu_caps)
          # Take the first GPU's compute capability (e.g. "7.5" or "8.9")
          string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _cap_match "${_gpu_caps}")
          set(_sm_major "${CMAKE_MATCH_1}")
          set(_sm_minor "${CMAKE_MATCH_2}")
          message(STATUS "Detected GPU: SM ${_sm_major}.${_sm_minor}")
          if(_sm_major GREATER 7 OR (_sm_major EQUAL 7 AND _sm_minor GREATER_EQUAL 5))
            set(_trt_sm_ok TRUE)
          else()
            message(STATUS "SM ${_sm_major}.${_sm_minor} < 7.5 — TensorRT 10.x not supported, using CUDA backend")
          endif()
        endif()
      endif()
      if(NOT _trt_sm_ok AND NOT NVIDIA_SMI)
        # No nvidia-smi (cross-compile or headless) — assume TensorRT OK
        set(_trt_sm_ok TRUE)
        message(STATUS "No nvidia-smi found — assuming TensorRT compatible (override with -DMINIGO_BACKEND=cuda)")
      endif()

      if(_trt_sm_ok)
        find_library(NVINFER_LIB nvinfer
          PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
                /usr/lib/aarch64-linux-gnu
          PATH_SUFFIXES lib lib64)
        find_path(NVINFER_INCLUDE_DIR NvInfer.h
          PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
                /usr/include/aarch64-linux-gnu)
        find_library(NVONNXPARSER_LIB nvonnxparser
          PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
                /usr/lib/aarch64-linux-gnu
          PATH_SUFFIXES lib lib64)
        if(NVINFER_LIB AND NVINFER_INCLUDE_DIR AND NVONNXPARSER_LIB)
          set(MINIGO_BACKEND "tensorrt")
        else()
          set(MINIGO_BACKEND "cuda")
        endif()
      else()
        set(MINIGO_BACKEND "cuda")
      endif()
    else()
      find_package(OpenCL QUIET)
      if(OpenCL_FOUND)
        set(MINIGO_BACKEND "opencl")
      else()
        set(MINIGO_BACKEND "eigen")
      endif()
    endif()
  endif()
endif()

# ── Default CUDA architectures (shared by cuda and tensorrt backends) ──
# Turing(75) Ampere(80,86) Ada(89) SASS + Hopper(90) PTX.
# The 90-virtual PTX JIT-compiles on Hopper, Blackwell, and future GPUs.
# Override with: cmake .. -DCMAKE_CUDA_ARCHITECTURES=89  (for faster local builds)
# Note: only affects .cu files (CUDA backend); TensorRT has no .cu files.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
  set(CMAKE_CUDA_ARCHITECTURES "75-real;80-real;86-real;89-real;90-virtual")
endif()

# ── Validate and configure selected backend ─────────────────
set(HAS_OPENCL FALSE)
set(HAS_METAL FALSE)
set(HAS_CUDA FALSE)
set(HAS_TENSORRT FALSE)
set(HAS_EIGEN FALSE)

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

elseif(MINIGO_BACKEND STREQUAL "tensorrt")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  find_library(NVINFER_LIB nvinfer REQUIRED
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
          /usr/lib/aarch64-linux-gnu
    PATH_SUFFIXES lib lib64)
  find_path(NVINFER_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu /usr/include /usr/local/include
          /usr/include/aarch64-linux-gnu
    REQUIRED)
  find_library(NVONNXPARSER_LIB nvonnxparser REQUIRED
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib /usr/local/lib
          /usr/lib/aarch64-linux-gnu
    PATH_SUFFIXES lib lib64)
  set(HAS_TENSORRT TRUE)

elseif(MINIGO_BACKEND STREQUAL "cuda")
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(HAS_CUDA TRUE)

elseif(MINIGO_BACKEND STREQUAL "opencl")
  find_package(OpenCL REQUIRED)
  set(HAS_OPENCL TRUE)

elseif(MINIGO_BACKEND STREQUAL "eigen")
  set(HAS_EIGEN TRUE)

else()
  message(FATAL_ERROR "Unknown backend: ${MINIGO_BACKEND}. Use: auto, metal, tensorrt, cuda, opencl, eigen")
endif()

# ── Library: minigo core ────────────────────────────────────
set(CORE_SOURCES
    src/game.cpp
    src/onnx_loader.cpp
    src/loaded_model.cpp
    src/compute_context.cpp
    src/nn_evaluator.cpp
    src/mcts.cpp
)

if(HAS_EIGEN)
  list(APPEND CORE_SOURCES src/eigen_compute.cpp)
endif()

if(HAS_OPENCL)
  list(APPEND CORE_SOURCES src/opencl_compute.cpp)
endif()

if(HAS_CUDA)
  list(APPEND CORE_SOURCES src/cuda_compute.cu)
endif()

if(HAS_TENSORRT)
  list(APPEND CORE_SOURCES src/tensorrt_compute.cpp)
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

if(HAS_EIGEN)
  target_compile_definitions(minigo_core PUBLIC MINIGO_HAS_EIGEN=1)
endif()

if(HAS_OPENCL)
  target_compile_definitions(minigo_core PUBLIC MINIGO_HAS_OPENCL=1)
  target_link_libraries(minigo_core PUBLIC OpenCL::OpenCL)
endif()

if(HAS_CUDA)
  target_compile_definitions(minigo_core PUBLIC MINIGO_HAS_CUDA=1)
  set_target_properties(minigo_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

if(HAS_TENSORRT)
  target_compile_definitions(minigo_core PUBLIC MINIGO_HAS_TENSORRT=1)
  target_include_directories(minigo_core PUBLIC
    ${NVINFER_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  find_library(CUDART_LIB cudart
    PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
    REQUIRED)
  target_link_libraries(minigo_core PUBLIC ${NVINFER_LIB} ${NVONNXPARSER_LIB} ${CUDART_LIB})
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
add_executable(evaluate  src/main_evaluate.cpp)

# Prefer ncursesw for wide-character (Unicode) support
find_library(NCURSESW_LIB ncursesw)
if(NCURSESW_LIB)
  target_link_libraries(play PRIVATE minigo_core ${NCURSESW_LIB})
else()
  find_package(Curses REQUIRED)
  target_link_libraries(play PRIVATE minigo_core ${CURSES_LIBRARIES})
  target_include_directories(play PRIVATE ${CURSES_INCLUDE_DIR})
endif()
target_link_libraries(selfplay  PRIVATE minigo_core)
target_link_libraries(benchmark PRIVATE minigo_core)
target_link_libraries(evaluate  PRIVATE minigo_core)

# ── Summary ─────────────────────────────────────────────────
message(STATUS "System:     ${CMAKE_SYSTEM_NAME} ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Compiler:   ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Backend:    ${MINIGO_BACKEND}")
message(STATUS "Eigen3:     ${EIGEN3_INCLUDE_DIR}")
