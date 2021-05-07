cmake_minimum_required(VERSION 3.5)
########################################
# Borrowed and adapted from TVM project
########################################
project(dgl C CXX)
message(STATUS "Start configuring project ${PROJECT_NAME}")

# cmake utils
include(cmake/util/Util.cmake)
include(cmake/util/MshadowUtil.cmake)
include(cmake/util/FindCUDA.cmake)

if(EXISTS ${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
  include(${CMAKE_CURRENT_BINARY_DIR}/config.cmake)
else()
  if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/config.cmake)
  endif()
endif()

# NOTE: do not modify this file to change option values.
# You can create a config.cmake at build folder
# and add set(OPTION VALUE) to override these build options.
# Alernatively, use cmake -DOPTION=VALUE through command-line.
dgl_option(USE_CUDA "Build with CUDA" OFF)
dgl_option(USE_OPENMP "Build with OpenMP" ON)
dgl_option(USE_AVX "Build with AVX optimization" ON)
dgl_option(USE_FP16 "Build with fp16 support to enable mixed precision training" OFF)
dgl_option(USE_TVM "Build with TVM kernels" OFF)
dgl_option(BUILD_CPP_TEST "Build cpp unittest executables" OFF)
dgl_option(LIBCXX_ENABLE_PARALLEL_ALGORITHMS "Enable the parallel algorithms library. This requires the PSTL to be available." OFF)
dgl_option(USE_S3 "Build with S3 support" OFF)
dgl_option(USE_HDFS "Build with HDFS support" OFF) # Set env HADOOP_HDFS_HOME if needed

# Set debug compile option for gdb, only happens when -DCMAKE_BUILD_TYPE=DEBUG
if (NOT MSVC)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g3 -ggdb")
endif(NOT MSVC)

if(USE_CUDA)
  message(STATUS "Build with CUDA support")
  project(dgl C CXX)
  include(cmake/modules/CUDA.cmake)
  if ((CUDA_VERSION_MAJOR LESS 11) OR
      ((CUDA_VERSION_MAJOR EQUAL 11) AND (CUDA_VERSION_MINOR EQUAL 0)))
    # For cuda<11, use external CUB/Thrust library because CUB is not part of CUDA.
    # For cuda==11.0, use external CUB/Thrust library because there is a bug in the
    #   official CUB library which causes invalid device ordinal error for DGL. The bug
    #   is fixed by https://github.com/NVIDIA/cub/commit/9143e47e048641aa0e6ddfd645bcd54ff1059939
    #   in 11.1.
    message(STATUS "Detected CUDA of version ${CUDA_VERSION}. Use external CUB/Thrust library.")
    cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/thrust")
    cuda_include_directories(BEFORE "${CMAKE_SOURCE_DIR}/third_party/cub")
  endif()
endif(USE_CUDA)

# initial variables
if(NOT MSVC)
set(DGL_LINKER_LIBS "dl")
endif(NOT MSVC)

if(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set(DGL_RUNTIME_LINKER_LIBS "")
else(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")
set(DGL_RUNTIME_LINKER_LIBS "rt")
endif(MSVC OR CMAKE_SYSTEM_NAME STREQUAL "Darwin")

# Generic compilation options
if(MSVC)
  add_definitions(-DWIN32_LEAN_AND_MEAN)
  add_definitions(-D_CRT_SECURE_NO_WARNINGS)
  add_definitions(-D_SCL_SECURE_NO_WARNINGS)
  add_definitions(-DNOMINMAX)
  set(CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS 1)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /bigobj")
  if(USE_MSVC_MT)
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()
else(MSVC)
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++11"    SUPPORT_CXX11)
  set(CMAKE_C_FLAGS "-O2 -Wall -fPIC ${CMAKE_C_FLAGS}")
  # We still use c++11 flag in CPU build because gcc5.4 (our default compiler) is
  # not fully compatible with c++14 feature.
  set(CMAKE_CXX_FLAGS "-O2 -Wall -fPIC -std=c++11 ${CMAKE_CXX_FLAGS}")
  if(NOT APPLE)
    set(CMAKE_SHARED_LINKER_FLAGS "-Wl,--warn-common ${CMAKE_SHARED_LINKER_FLAGS}")
  endif(NOT APPLE)
endif(MSVC)

if(USE_OPENMP)
  include(FindOpenMP)
  if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_CXX_FLAGS}")
  endif(OPENMP_FOUND)
  message(STATUS "Build with OpenMP.")
endif(USE_OPENMP)

if(USE_AVX)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_AVX")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_AVX")
  message(STATUS "Build with AVX optimization.")
endif(USE_AVX)

# Build with fp16 to support mixed precision training.
if(USE_FP16)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_FP16")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_FP16")
  message(STATUS "Build with fp16 to support mixed precision training")
endif(USE_FP16)

# To compile METIS correct for DGL.
if(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} /DIDXTYPEWIDTH=64 /DREALTYPEWIDTH=32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DIDXTYPEWIDTH=64 /DREALTYPEWIDTH=32")
else(MSVC)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DIDXTYPEWIDTH=64 -DREALTYPEWIDTH=32")
endif(MSVC)

# configure minigun
add_definitions(-DENABLE_PARTIAL_FRONTIER=0)  # disable minigun partial frontier compile
# Source file lists
file(GLOB DGL_SRC
  src/*.cc
  src/array/*.cc
  src/array/cpu/*.cc
  src/random/*.cc
  src/random/cpu/*.cc
  src/kernel/*.cc
  src/kernel/cpu/*.cc
  src/runtime/*.cc
  src/geometry/*.cc
  src/geometry/cpu/*.cc
  src/dataloading/*.cc
)

file(GLOB_RECURSE DGL_SRC_1
  src/api/*.cc
  src/graph/*.cc
  src/scheduler/*.cc
  src/rpc/*.cc
)

list(APPEND DGL_SRC ${DGL_SRC_1})

# Configure cuda
if(USE_CUDA)
  dgl_config_cuda(DGL_CUDA_SRC)
  list(APPEND DGL_SRC ${DGL_CUDA_SRC})
endif(USE_CUDA)

if(USE_CUDA)
  cuda_add_library(dgl SHARED ${DGL_SRC})
else(USE_CUDA)
  add_library(dgl SHARED ${DGL_SRC} src/graph/sampling/randomwalks/node2vec_impl.h)
endif(USE_CUDA)

# include directories
target_include_directories(dgl PRIVATE "include")
target_include_directories(dgl PRIVATE "third_party/dlpack/include")
target_include_directories(dgl PRIVATE "third_party/dmlc-core/include")
target_include_directories(dgl PRIVATE "third_party/minigun/minigun")
target_include_directories(dgl PRIVATE "third_party/minigun/third_party/moderngpu/src")
target_include_directories(dgl PRIVATE "third_party/phmap/")
target_include_directories(dgl PRIVATE "third_party/xbyak/")
target_include_directories(dgl PRIVATE "third_party/METIS/include/")
target_include_directories(dgl PRIVATE "tensoradapter/include")
target_include_directories(dgl PRIVATE "third_party/nanoflann/include")

# For serialization
if (USE_HDFS)
  option(DMLC_HDFS_SHARED "dgl has to build with dynamic hdfs library" ON)
endif()
add_subdirectory("third_party/dmlc-core")
list(APPEND DGL_LINKER_LIBS dmlc)
set(GOOGLE_TEST 0) # Turn off dmlc-core test

# Compile METIS
if(NOT MSVC)
  set(GKLIB_PATH "${CMAKE_SOURCE_DIR}/third_party/METIS/GKlib")
  include(${GKLIB_PATH}/GKlibSystem.cmake)
  include_directories(${GKLIB_PATH})
  include_directories("third_party/METIS/include/")
  add_subdirectory("third_party/METIS/libmetis/")
  list(APPEND DGL_LINKER_LIBS metis)
endif(NOT MSVC)

# Compile TVM Runtime and Featgraph
# (NOTE) We compile a dynamic library called featgraph_runtime, which the DGL library links to.
# Kernels are packed in a separate dynamic library called featgraph_kernels, which DGL
# will load during runtime.
if(USE_TVM)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DUSE_TVM")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_TVM")
  target_include_directories(dgl PRIVATE "featgraph/include")
  add_subdirectory("featgraph/")
  list(APPEND DGL_LINKER_LIBS featgraph_runtime)
  message(STATUS "Build with TVM runtime and featgraph kernels.")
endif(USE_TVM)

# support PARALLEL_ALGORITHMS
if (LIBCXX_ENABLE_PARALLEL_ALGORITHMS)
  add_definitions(-DPARALLEL_ALGORITHMS)
endif(LIBCXX_ENABLE_PARALLEL_ALGORITHMS)

target_link_libraries(dgl ${DGL_LINKER_LIBS} ${DGL_RUNTIME_LINKER_LIBS})
if(MSVC)
  add_custom_command(
    TARGET dgl POST_BUILD COMMAND
    cmd.exe /c "COPY /Y Release\\dgl.dll .")
endif(MSVC)

# Tensor adapter libraries
# Linking against LibTorch involves linking against a bunch of other libraries
# returned by PyTorch's CMake (e.g. C10 or NVTools).  Because CMake caches
# the found libraries in find_library(), often times CMake will look into the libraries
# of the wrong version when I build everything in the same CMake process.  As
# a result, I (BarclayII) am launching an individual CMake build for every PyTorch version.
if(BUILD_TORCH)
  file(TO_NATIVE_PATH ${CMAKE_CURRENT_BINARY_DIR} BINDIR)
  file(TO_NATIVE_PATH ${CMAKE_COMMAND} CMAKE_CMD)
  if(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tensoradapter/pytorch/build.bat BUILD_SCRIPT)
    add_custom_target(
      tensoradapter_pytorch
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
      BINDIR=${BINDIR}
      cmd /e:on /c ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensoradapter/pytorch)
  else(MSVC)
    file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tensoradapter/pytorch/build.sh BUILD_SCRIPT)
    add_custom_target(
      tensoradapter_pytorch
      ${CMAKE_COMMAND} -E env
      CMAKE_COMMAND=${CMAKE_CMD}
      CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR}
      BINDIR=${CMAKE_CURRENT_BINARY_DIR}
      bash ${BUILD_SCRIPT} ${TORCH_PYTHON_INTERPS}
      DEPENDS ${BUILD_SCRIPT}
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/tensoradapter/pytorch)
  endif(MSVC)
  add_dependencies(dgl tensoradapter_pytorch)
endif(BUILD_TORCH)

# Installation rules
install(TARGETS dgl DESTINATION lib${LIB_SUFFIX})

# Testing
if(BUILD_CPP_TEST)
  message(STATUS "Build with unittest")
  add_subdirectory(./third_party/googletest)
  enable_testing()
  include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})
  include_directories("include")
  include_directories("third_party/dlpack/include")
  include_directories("third_party/xbyak")
  include_directories("third_party/dmlc-core/include")
  include_directories("third_party/phmap")
  file(GLOB_RECURSE TEST_SRC_FILES ${PROJECT_SOURCE_DIR}/tests/cpp/*.cc)
  add_executable(runUnitTests ${TEST_SRC_FILES})
  target_link_libraries(runUnitTests gtest gtest_main)
  target_link_libraries(runUnitTests dgl)
  add_test(UnitTests runUnitTests)
endif(BUILD_CPP_TEST)
