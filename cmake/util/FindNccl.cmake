#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Tries to find NCCL headers and libraries.
#
# Usage of this module as follows:
#
#  find_package(NCCL)
#
# Variables used by this module, they can change the default behaviour and need
# to be set before calling find_package:
#
#  NCCL_ROOT - When set, this path is inspected instead of standard library
#              locations as the root of the NCCL installation.
#              The environment variable NCCL_ROOT overrides this variable.
#
# This module defines
#  Nccl_FOUND, whether nccl has been found
#  NCCL_INCLUDE_DIR, directory containing header
#  NCCL_LIBRARY, directory containing nccl library
#  NCCL_LIB_NAME, nccl library name
#  USE_NCCL_LIB_PATH, when set, NCCL_LIBRARY path is also inspected for the 
#                     location of the nccl library. This would disable
#                     switching between static and shared.
#
# This module assumes that the user has already called find_package(CUDA)
#
# This file is from https://github.com/dmlc/xgboost, with modifications to
# check the version.

if (NCCL_LIBRARY)
  if(NOT USE_NCCL_LIB_PATH)
    # Don't cache NCCL_LIBRARY to enable switching between static and shared.
    unset(NCCL_LIBRARY CACHE)
  endif(NOT USE_NCCL_LIB_PATH)
endif()

if (BUILD_WITH_SHARED_NCCL)
  # libnccl.so
  set(NCCL_LIB_NAME nccl)
else ()
  # libnccl_static.a
  set(NCCL_LIB_NAME nccl_static)
endif (BUILD_WITH_SHARED_NCCL)

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  PATHS $ENV{NCCL_ROOT}/include ${NCCL_ROOT}/include)

# make sure it has point to point support
file(STRINGS "${NCCL_INCLUDE_DIR}/nccl.h" NCCL_VERSION_CODE REGEX "^#define[ \t]+NCCL_VERSION_CODE[ \t]+[0-9]+.*$" LIMIT_COUNT 1)
string(REGEX REPLACE "^.*NCCL_VERSION_CODE[ \t]+([0-9]+).*$" "\\1" NCCL_VERSION "${NCCL_VERSION_CODE}")


find_library(NCCL_LIBRARY
  NAMES ${NCCL_LIB_NAME}
  PATHS $ENV{NCCL_ROOT}/lib/ ${NCCL_ROOT}/lib)

if ("${NCCL_VERSION}" LESS "2700")
  message(FATAL_ERROR "Require nccl >= 2700, but found ${NCCL_LIBRARY}==${NCCL_VERSION}")
else()
  message(STATUS "Using nccl library: ${NCCL_LIBRARY} ${NCCL_VERSION}")
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Nccl DEFAULT_MSG
                                  NCCL_INCLUDE_DIR NCCL_LIBRARY)

mark_as_advanced(
  NCCL_INCLUDE_DIR
  NCCL_LIBRARY
)
