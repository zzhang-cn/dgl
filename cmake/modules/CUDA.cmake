# CUDA Module
if(USE_CUDA)
  find_cuda(${USE_CUDA} REQUIRED)
else(USE_CUDA)
  return()
endif()

###### Borrowed from MSHADOW project

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-std=c++14"   SUPPORT_CXX14)

set(dgl_known_gpu_archs "35 50 60 70 75 80")

################################################################################################
# A function for automatic detection of GPUs installed  (if autodetection is enabled)
# Usage:
#   dgl_detect_installed_gpus(out_variable)
function(dgl_detect_installed_gpus out_variable)
set(CUDA_gpu_detect_output "")
  if(NOT CUDA_gpu_detect_output)
    message(STATUS "Running GPU architecture autodetection")
    set(__cufile ${PROJECT_BINARY_DIR}/detect_cuda_archs.cu)

    file(WRITE ${__cufile} ""
      "#include <cstdio>\n"
      "#include <iostream>\n"
      "using namespace std;\n"
      "int main()\n"
      "{\n"
      "  int count = 0;\n"
      "  if (cudaSuccess != cudaGetDeviceCount(&count)) { return -1; }\n"
      "  if (count == 0) { cerr << \"No cuda devices detected\" << endl; return -1; }\n"
      "  for (int device = 0; device < count; ++device)\n"
      "  {\n"
      "    cudaDeviceProp prop;\n"
      "    if (cudaSuccess == cudaGetDeviceProperties(&prop, device))\n"
      "      std::printf(\"%d.%d \", prop.major, prop.minor);\n"
      "  }\n"
      "  return 0;\n"
      "}\n")
    if(MSVC)
      #find vcvarsall.bat and run it building msvc environment
      get_filename_component(MY_COMPILER_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)
      find_file(MY_VCVARSALL_BAT vcvarsall.bat "${MY_COMPILER_DIR}/.." "${MY_COMPILER_DIR}/../..")
      execute_process(COMMAND ${MY_VCVARSALL_BAT} && ${CUDA_NVCC_EXECUTABLE} -arch sm_35 --run  ${__cufile}
                      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                      RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
    else()
      if(CUDA_LIBRARY_PATH)
        set(CUDA_LINK_LIBRARY_PATH "-L${CUDA_LIBRARY_PATH}")
      endif()
      execute_process(COMMAND ${CUDA_NVCC_EXECUTABLE} -arch sm_35 --run ${__cufile} ${CUDA_LINK_LIBRARY_PATH}
                      WORKING_DIRECTORY "${PROJECT_BINARY_DIR}/CMakeFiles/"
                      RESULT_VARIABLE __nvcc_res OUTPUT_VARIABLE __nvcc_out
                      OUTPUT_STRIP_TRAILING_WHITESPACE)
    endif()
    if(__nvcc_res EQUAL 0)
      # nvcc outputs text containing line breaks when building with MSVC.
      # The line below prevents CMake from inserting a variable with line
      # breaks in the cache
      message(STATUS "Found CUDA arch ${__nvcc_out}")
      string(REGEX MATCH "([1-9].[0-9])" __nvcc_out "${__nvcc_out}")
      string(REPLACE "2.1" "2.1(2.0)" __nvcc_out "${__nvcc_out}")
      set(CUDA_gpu_detect_output ${__nvcc_out} CACHE INTERNAL "Returned GPU architetures from mshadow_detect_gpus tool" FORCE)
    else()
      message(WARNING "Running GPU detection script with nvcc failed: ${__nvcc_out}")
    endif()
  endif()

  if(NOT CUDA_gpu_detect_output)
    message(WARNING "Automatic GPU detection failed. Building for all known architectures (${dgl_known_gpu_archs}).")
    set(${out_variable} ${dgl_known_gpu_archs} PARENT_SCOPE)
  else()
    set(${out_variable} ${CUDA_gpu_detect_output} PARENT_SCOPE)
  endif()
endfunction()


################################################################################################
# Function for selecting GPU arch flags for nvcc based on CUDA_ARCH_NAME
# Usage:
#   dgl_select_nvcc_arch_flags(out_variable)
function(dgl_select_nvcc_arch_flags out_variable)
  # List of arch names
  set(__archs_names "Fermi" "Kepler" "Maxwell" "Pascal" "Volta" "Turing" "Ampere" "All" "Manual")
  set(__archs_name_default "All")
  if(NOT CMAKE_CROSSCOMPILING)
    list(APPEND __archs_names "Auto")
    set(__archs_name_default "Auto")
  endif()

  # set CUDA_ARCH_NAME strings (so it will be seen as dropbox in CMake-Gui)
  set(CUDA_ARCH_NAME ${__archs_name_default} CACHE STRING "Select target NVIDIA GPU achitecture.")
  set_property( CACHE CUDA_ARCH_NAME PROPERTY STRINGS "" ${__archs_names} )
  mark_as_advanced(CUDA_ARCH_NAME)

  # verify CUDA_ARCH_NAME value
  if(NOT ";${__archs_names};" MATCHES ";${CUDA_ARCH_NAME};")
    string(REPLACE ";" ", " __archs_names "${__archs_names}")
    message(FATAL_ERROR "Only ${__archs_names} architeture names are supported.")
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(CUDA_ARCH_BIN ${dgl_known_gpu_archs} CACHE STRING "Specify 'real' GPU architectures to build binaries for, BIN(PTX) format is supported")
    set(CUDA_ARCH_PTX "50"                     CACHE STRING "Specify 'virtual' PTX architectures to build PTX intermediate code for")
    mark_as_advanced(CUDA_ARCH_BIN CUDA_ARCH_PTX)
  else()
    unset(CUDA_ARCH_BIN CACHE)
    unset(CUDA_ARCH_PTX CACHE)
  endif()

  if(${CUDA_ARCH_NAME} STREQUAL "Fermi")
    set(__cuda_arch_bin "20 21(20)")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Kepler")
    set(__cuda_arch_bin "30 35")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Maxwell")
    set(__cuda_arch_bin "50")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Pascal")
    set(__cuda_arch_bin "60 61")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Volta")
    set(__cuda_arch_bin "70")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Turing")
    set(__cuda_arch_bin "75")
  elseif(${CUDA_ARCH_NAME} STREQUAL "Ampere")
    set(__cuda_arch_bin "80")
  elseif(${CUDA_ARCH_NAME} STREQUAL "All")
    set(__cuda_arch_bin ${dgl_known_gpu_archs})
  elseif(${CUDA_ARCH_NAME} STREQUAL "Auto")
    dgl_detect_installed_gpus(__cuda_arch_bin)
  else()  # (${CUDA_ARCH_NAME} STREQUAL "Manual")
    set(__cuda_arch_bin ${CUDA_ARCH_BIN})
  endif()

  # remove dots and convert to lists
  string(REGEX REPLACE "\\." "" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX REPLACE "\\." "" __cuda_arch_ptx "${CUDA_ARCH_PTX}")
  string(REGEX MATCHALL "[0-9()]+" __cuda_arch_bin "${__cuda_arch_bin}")
  string(REGEX MATCHALL "[0-9]+"   __cuda_arch_ptx "${__cuda_arch_ptx}")
  mshadow_list_unique(__cuda_arch_bin __cuda_arch_ptx)

  set(__nvcc_flags "--expt-relaxed-constexpr")
  set(__nvcc_archs_readable "")

  # Tell NVCC to add binaries for the specified GPUs
  foreach(__arch ${__cuda_arch_bin})
    if(__arch MATCHES "([0-9]+)\\(([0-9]+)\\)")
      # User explicitly specified PTX for the concrete BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${CMAKE_MATCH_2},code=sm_${CMAKE_MATCH_1})
      list(APPEND __nvcc_archs_readable sm_${CMAKE_MATCH_1})
    else()
      # User didn't explicitly specify PTX for the concrete BIN, we assume PTX=BIN
      list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=sm_${__arch})
      list(APPEND __nvcc_archs_readable sm_${__arch})
    endif()
  endforeach()

  # Tell NVCC to add PTX intermediate code for the specified architectures
  foreach(__arch ${__cuda_arch_ptx})
    list(APPEND __nvcc_flags -gencode arch=compute_${__arch},code=compute_${__arch})
    list(APPEND __nvcc_archs_readable compute_${__arch})
  endforeach()

  string(REPLACE ";" " " __nvcc_archs_readable "${__nvcc_archs_readable}")
  set(${out_variable}          ${__nvcc_flags}          PARENT_SCOPE)
  set(${out_variable}_readable ${__nvcc_archs_readable} PARENT_SCOPE)
endfunction()

################################################################################################
# Short command for cuda comnpilation
# Usage:
#   dgl_cuda_compile(<objlist_variable> <cuda_files>)
macro(dgl_cuda_compile objlist_variable)
  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var}_backup_in_cuda_compile_ "${${var}}")

    # we remove /EHa as it generates warnings under windows
    string(REPLACE "/EHa" "" ${var} "${${var}}")

  endforeach()
  if(UNIX OR APPLE)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -fPIC --std=c++14)
  endif()

  if(APPLE)
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler -Wno-unused-function)
  endif()

  set(CUDA_NVCC_FLAGS_DEBUG "${CUDA_NVCC_FLAGS_DEBUG} -G")

  if(MSVC)
    # disable noisy warnings:
    # 4819: The file contains a character that cannot be represented in the current code page (number).
    list(APPEND CUDA_NVCC_FLAGS -Xcompiler "/wd4819")
    foreach(flag_var
        CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_DEBUG CMAKE_CXX_FLAGS_RELEASE
        CMAKE_CXX_FLAGS_MINSIZEREL CMAKE_CXX_FLAGS_RELWITHDEBINFO)
      if(${flag_var} MATCHES "/MD")
        string(REGEX REPLACE "/MD" "/MT" ${flag_var} "${${flag_var}}")
      endif(${flag_var} MATCHES "/MD")
    endforeach(flag_var)
  endif()

  # If the build system is a container, make sure the nvcc intermediate files
  # go into the build output area rather than in /tmp, which may run out of space
  if(IS_CONTAINER_BUILD)
    set(CUDA_NVCC_INTERMEDIATE_DIR "${CMAKE_CURRENT_BINARY_DIR}")
    message(STATUS "Container build enabled, so nvcc intermediate files in: ${CUDA_NVCC_INTERMEDIATE_DIR}")
    list(APPEND CUDA_NVCC_FLAGS "--keep --keep-dir ${CUDA_NVCC_INTERMEDIATE_DIR}")
  endif()

  cuda_compile(cuda_objcs ${ARGN})

  foreach(var CMAKE_CXX_FLAGS CMAKE_CXX_FLAGS_RELEASE CMAKE_CXX_FLAGS_DEBUG)
    set(${var} "${${var}_backup_in_cuda_compile_}")
    unset(${var}_backup_in_cuda_compile_)
  endforeach()

  set(${objlist_variable} ${cuda_objcs})
endmacro()

################################################################################################
# Config cuda compilation.
# Usage:
#   dgl_config_cuda(<dgl_cuda_src>)
macro(dgl_config_cuda out_variable)
  if(NOT CUDA_FOUND)
    message(FATAL_ERROR "Cannot find CUDA.")
  endif()
  # always set the includedir when cuda is available
  # avoid global retrigger of cmake
	include_directories(${CUDA_INCLUDE_DIRS})

  add_definitions(-DDGL_USE_CUDA)

  file(GLOB_RECURSE DGL_CUDA_SRC
    src/array/cuda/*.cc
    src/array/cuda/*.cu
    src/kernel/cuda/*.cc
    src/kernel/cuda/*.cu
    src/runtime/cuda/*.cc
    src/runtime/cuda/*.cu
    src/geometry/cuda/*.cu
    src/graph/transform/cuda/*.cu
  )

  # NVCC flags
  # Manually set everything
  set(CUDA_PROPAGATE_HOST_FLAGS OFF)

  # 0. Add host flags
  message(STATUS "${CMAKE_CXX_FLAGS}")
  string(REGEX REPLACE "[ \t\n\r]" "," CXX_HOST_FLAGS "${CMAKE_CXX_FLAGS}")
  if(MSVC AND NOT USE_MSVC_MT)
    string(CONCAT CXX_HOST_FLAGS ${CXX_HOST_FLAGS} ",/MD")
  endif()
  list(APPEND CUDA_NVCC_FLAGS "-Xcompiler ,${CXX_HOST_FLAGS}")

  # 1. Add arch flags
  dgl_select_nvcc_arch_flags(NVCC_FLAGS_ARCH)
  list(APPEND CUDA_NVCC_FLAGS ${NVCC_FLAGS_ARCH})

  # 2. flags in third_party/moderngpu
  list(APPEND CUDA_NVCC_FLAGS "--expt-extended-lambda;-Wno-deprecated-declarations")


  # 3. CUDA 11 requires c++14 by default
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("-std=c++14"    SUPPORT_CXX14)
  string(REPLACE "-std=c++11" "" CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")
  list(APPEND CUDA_NVCC_FLAGS "-std=c++14")

  message(STATUS "CUDA flags: ${CUDA_NVCC_FLAGS}")

  list(APPEND DGL_LINKER_LIBS
    ${CUDA_CUDART_LIBRARY}
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_cusparse_LIBRARY}
    ${CUDA_CURAND_LIBRARY})

  set(${out_variable} ${DGL_CUDA_SRC})
endmacro()
