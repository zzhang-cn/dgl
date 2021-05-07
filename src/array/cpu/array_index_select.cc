/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/array_index_select.cc
 * \brief Array index select CPU implementation
 */
#include <dgl/array.h>

namespace dgl {
using runtime::NDArray;
namespace aten {
namespace impl {

template<DLDeviceType XPU, typename DType, typename IdType>
NDArray IndexSelect(NDArray array, IdArray index) {
  CHECK_EQ(array->shape[0], array.NumElements()) << "Only support tensor"
    << " whose first dimension equals number of elements, e.g. (5,), (5, 1)";

  const DType* array_data = static_cast<DType*>(array->data);
  const IdType* idx_data = static_cast<IdType*>(index->data);
  const int64_t arr_len = array->shape[0];
  const int64_t len = index->shape[0];
  NDArray ret = NDArray::Empty({len}, array->dtype, array->ctx);
  DType* ret_data = static_cast<DType*>(ret->data);
  for (int64_t i = 0; i < len; ++i) {
    CHECK_LT(idx_data[i], arr_len) << "Index out of range.";
    ret_data[i] = array_data[idx_data[i]];
  }
  return ret;
}

template NDArray IndexSelect<kDLCPU, int32_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLCPU, int32_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLCPU, int64_t, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLCPU, int64_t, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLCPU, float, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLCPU, float, int64_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLCPU, double, int32_t>(NDArray, IdArray);
template NDArray IndexSelect<kDLCPU, double, int64_t>(NDArray, IdArray);

template <DLDeviceType XPU, typename DType>
DType IndexSelect(NDArray array, int64_t index) {
  const DType* data = static_cast<DType*>(array->data);
  return data[index];
}

template int32_t IndexSelect<kDLCPU, int32_t>(NDArray array, int64_t index);
template int64_t IndexSelect<kDLCPU, int64_t>(NDArray array, int64_t index);
template float IndexSelect<kDLCPU, float>(NDArray array, int64_t index);
template double IndexSelect<kDLCPU, double>(NDArray array, int64_t index);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
