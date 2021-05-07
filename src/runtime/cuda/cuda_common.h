/*!
 *  Copyright (c) 2017 by Contributors
 * \file cuda_common.h
 * \brief Common utilities for CUDA
 */
#ifndef DGL_RUNTIME_CUDA_CUDA_COMMON_H_
#define DGL_RUNTIME_CUDA_CUDA_COMMON_H_

#include <cublas_v2.h>
#include <cusparse.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <dgl/runtime/packed_func.h>
#include <string>
#include "../workspace_pool.h"

namespace dgl {
namespace runtime {

template <typename T>
inline bool is_zero(T size) {
    return size == 0;
}

template <>
inline bool is_zero<dim3>(dim3 size) {
    return size.x == 0 || size.y == 0 || size.z == 0;
}

#define CUDA_DRIVER_CALL(x)                                             \
  {                                                                     \
    CUresult result = x;                                                \
    if (result != CUDA_SUCCESS && result != CUDA_ERROR_DEINITIALIZED) { \
      const char *msg;                                                  \
      cuGetErrorName(result, &msg);                                     \
      LOG(FATAL)                                                        \
          << "CUDAError: " #x " failed with error: " << msg;            \
    }                                                                   \
  }

#define CUDA_CALL(func)                                            \
  {                                                                \
    cudaError_t e = (func);                                        \
    CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)       \
        << "CUDA: " << cudaGetErrorString(e);                      \
  }

#define CUDA_KERNEL_CALL(kernel, nblks, nthrs, shmem, stream, ...) \
  {                                                                \
    if (!dgl::runtime::is_zero((nblks)) &&                         \
        !dgl::runtime::is_zero((nthrs))) {                         \
      (kernel) <<< (nblks), (nthrs), (shmem), (stream) >>>         \
        (__VA_ARGS__);                                             \
      cudaError_t e = cudaGetLastError();                          \
      CHECK(e == cudaSuccess || e == cudaErrorCudartUnloading)     \
          << "CUDA kernel launch error: "                          \
          << cudaGetErrorString(e);                                \
    }                                                              \
  }

#define CUSPARSE_CALL(func)                                        \
  {                                                                \
    cusparseStatus_t e = (func);                                   \
    CHECK(e == CUSPARSE_STATUS_SUCCESS)                            \
        << "CUSPARSE ERROR: " << e;                                \
  }

#define CUBLAS_CALL(func)                                          \
  {                                                                \
    cublasStatus_t e = (func);                                     \
    CHECK(e == CUBLAS_STATUS_SUCCESS) << "CUBLAS ERROR: " << e;    \
  }

#define CURAND_CALL(func)                                           \
{                                                                   \
  curandStatus_t e = (func);                                        \
  CHECK(e == CURAND_STATUS_SUCCESS)                                 \
    << "CURAND Error: " << dgl::runtime::curandGetErrorString(e)    \
    << " at " << __FILE__ << ":" << __LINE__;                       \
}

inline const char* curandGetErrorString(curandStatus_t error) {
  switch (error) {
  case CURAND_STATUS_SUCCESS:
    return "CURAND_STATUS_SUCCESS";
  case CURAND_STATUS_VERSION_MISMATCH:
    return "CURAND_STATUS_VERSION_MISMATCH";
  case CURAND_STATUS_NOT_INITIALIZED:
    return "CURAND_STATUS_NOT_INITIALIZED";
  case CURAND_STATUS_ALLOCATION_FAILED:
    return "CURAND_STATUS_ALLOCATION_FAILED";
  case CURAND_STATUS_TYPE_ERROR:
    return "CURAND_STATUS_TYPE_ERROR";
  case CURAND_STATUS_OUT_OF_RANGE:
    return "CURAND_STATUS_OUT_OF_RANGE";
  case CURAND_STATUS_LENGTH_NOT_MULTIPLE:
    return "CURAND_STATUS_LENGTH_NOT_MULTIPLE";
  case CURAND_STATUS_DOUBLE_PRECISION_REQUIRED:
    return "CURAND_STATUS_DOUBLE_PRECISION_REQUIRED";
  case CURAND_STATUS_LAUNCH_FAILURE:
    return "CURAND_STATUS_LAUNCH_FAILURE";
  case CURAND_STATUS_PREEXISTING_FAILURE:
    return "CURAND_STATUS_PREEXISTING_FAILURE";
  case CURAND_STATUS_INITIALIZATION_FAILED:
    return "CURAND_STATUS_INITIALIZATION_FAILED";
  case CURAND_STATUS_ARCH_MISMATCH:
    return "CURAND_STATUS_ARCH_MISMATCH";
  case CURAND_STATUS_INTERNAL_ERROR:
    return "CURAND_STATUS_INTERNAL_ERROR";
  }
  // To suppress compiler warning.
  return "Unrecognized curand error string";
}

/*
 * \brief Cast data type to cudaDataType_t.
 */
template <typename T>
struct cuda_dtype {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};

template <>
struct cuda_dtype<half> {
  static constexpr cudaDataType_t value = CUDA_R_16F;
};

template <>
struct cuda_dtype<float> {
  static constexpr cudaDataType_t value = CUDA_R_32F;
};

template <>
struct cuda_dtype<double> {
  static constexpr cudaDataType_t value = CUDA_R_64F;
};

#if CUDART_VERSION >= 11000
/*
 * \brief Cast index data type to cusparseIndexType_t.
 */
template <typename T>
struct cusparse_idtype {
  static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_32I;
};

template <>
struct cusparse_idtype<int32_t> {
  static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_32I;
};

template <>
struct cusparse_idtype<int64_t> {
  static constexpr cusparseIndexType_t value = CUSPARSE_INDEX_64I;
};
#endif

inline void cusparse_create_descr(cusparseMatDescr_t* descr) {
  CUSPARSE_CALL(cusparseCreateMatDescr(descr));
  CUSPARSE_CALL(cusparseSetMatDiagType(*descr, CUSPARSE_DIAG_TYPE_NON_UNIT));
  CUSPARSE_CALL(cusparseSetMatIndexBase(*descr, CUSPARSE_INDEX_BASE_ZERO));
  CUSPARSE_CALL(cusparseSetMatType(*descr, CUSPARSE_MATRIX_TYPE_GENERAL));
}

/*! \brief Thread local workspace */
class CUDAThreadEntry {
 public:
  /*! \brief The cuda stream */
  cudaStream_t stream{nullptr};
  /*! \brief The cusparse handler */
  cusparseHandle_t cusparse_handle{nullptr};
  /*! \brief The cublas handler */
  cublasHandle_t cublas_handle{nullptr};
  /*! \brief The curand generator */
  curandGenerator_t curand_gen{nullptr};
  /*! \brief thread local pool*/
  WorkspacePool pool;
  /*! \brief constructor */
  CUDAThreadEntry();
  // get the threadlocal workspace
  static CUDAThreadEntry* ThreadLocal();
};
}  // namespace runtime
}  // namespace dgl
#endif  // DGL_RUNTIME_CUDA_CUDA_COMMON_H_
