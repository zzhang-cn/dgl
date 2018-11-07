/*!
 *  Copyright (c) 2018 by Contributors
 * \file c_api_common.h
 * \brief DGL C API common util functions
 */
#ifndef DGL_C_API_COMMON_H_
#define DGL_C_API_COMMON_H_

#include <dgl/runtime/ndarray.h>
#include <dgl/runtime/packed_func.h>
#include <dgl/runtime/registry.h>
#include <vector>

namespace dgl {

// Graph handler type
typedef void* GraphHandle;

/*!
 * \brief Convert the given DLTensor to DLManagedTensor.
 *
 * Return a temporary DLManagedTensor that does not own memory.
 */
DLManagedTensor* CreateTmpDLManagedTensor(
    const tvm::runtime::TVMArgValue& arg);

/*!
 * \brief Convert a vector of NDArray to PackedFunc.
 */
tvm::runtime::PackedFunc ConvertNDArrayVectorToPackedFunc(
    const std::vector<tvm::runtime::NDArray>& vec);

/*!
 * \brief Copy a vector to an int64_t NDArray.
 *
 * The element type of the vector must be convertible to int64_t.
 */
template<typename DType>
tvm::runtime::NDArray CopyVectorToNDArray(
    const std::vector<DType>& vec) {
  using tvm::runtime::NDArray;
  const int64_t len = vec.size();
  NDArray a = NDArray::Empty({len}, DLDataType{kDLInt, 64, 1}, DLContext{kDLCPU, 0});
  std::copy(vec.begin(), vec.end(), static_cast<int64_t*>(a->data));
  return a;
}

}  // namespace dgl

#endif  // DGL_C_API_COMMON_H_
