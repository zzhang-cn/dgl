/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_index_select.cu
 * \brief Array index select GPU implementation
 */
#include <dgl/array.h>
#include "../../runtime/cuda/cuda_common.h"
#include "./array_index_select.cuh"
#include "./utils.h"

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template<DLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect(NDArray array, IdArray index) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  int64_t num_feat = 1;
  std::vector<int64_t> shape{len};
  for (int d = 1; d < array->ndim; ++d) {
    num_feat *= array->shape[d];
    shape.emplace_back(array->shape[d]);
  }

  NDArray ret = NDArray::Empty(shape, array->dtype, array->ctx);
  if (len == 0)
    return ret;
  DType* ret_data = static_cast<DType*>(ret->data);

  if (num_feat == 1) {
      const int nt = cuda::FindNumThreads(len);
      const int nb = (len + nt - 1) / nt;
      CUDA_KERNEL_CALL(IndexSelectSingleKernel, nb, nt, 0, thr_entry->stream,
          array_data, idx_data, len, ret_data);
  } else {
      dim3 block(256, 1);
      while (static_cast<int64_t>(block.x) >= 2*num_feat) {
          block.x /= 2;
          block.y *= 2;
      }
      const dim3 grid((len+block.y-1)/block.y);
      CUDA_KERNEL_CALL(IndexSelectMultiKernel, grid, block, 0, thr_entry->stream,
          array_data, num_feat, idx_data, len, ret_data);
  }
  return ret;
}

template NDArray IndexSelect<kDLGPU, int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLGPU, int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLGPU, int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLGPU, int64_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLGPU, float, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLGPU, float, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLGPU, double, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLGPU, double, int64_t>(NDArray, IdArray);

template <DLDeviceType XPU, typename DType>
DType IndexSelect(NDArray array, int64_t index) {
  auto device = runtime::DeviceAPI::Get(array->ctx);
  DType ret = 0;
  device->CopyDataFromTo(
      static_cast<DType*>(array->data) + index, 0, &ret, 0,
      sizeof(DType), array->ctx, DLContext{kDLCPU, 0},
      array->dtype, nullptr);
  return ret;
}

template int32_t IndexSelect<kDLGPU, int32_t>(NDArray array, int64_t index);
template int64_t IndexSelect<kDLGPU, int64_t>(NDArray array, int64_t index);
template uint32_t IndexSelect<kDLGPU, uint32_t>(NDArray array, int64_t index);
template uint64_t IndexSelect<kDLGPU, uint64_t>(NDArray array, int64_t index);
template float IndexSelect<kDLGPU, float>(NDArray array, int64_t index);
template double IndexSelect<kDLGPU, double>(NDArray array, int64_t index);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
