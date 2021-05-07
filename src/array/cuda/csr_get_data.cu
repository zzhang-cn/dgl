/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cuda/csr_get_data.cu
 * \brief Retrieve entries of a CSR matrix
 */
#include <dgl/array.h>
#include <vector>
#include <unordered_set>
#include <numeric>
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetData(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, DType filler) {
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];

  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
    << "Invalid row and col id array.";

  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;

  const int64_t rstlen = std::max(rowlen, collen);
  IdArray rst = NDArray::Empty({rstlen}, weights->dtype, rows->ctx);
  if (rstlen == 0)
    return rst;

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const int nt = cuda::FindNumThreads(rstlen);
  const int nb = (rstlen + nt - 1) / nt;
  if (return_eids)
    BUG_IF_FAIL(DLDataTypeTraits<DType>::dtype == rows->dtype) <<
      "DType does not match row's dtype.";

  // TODO(minjie): use binary search for sorted csr
  CUDA_KERNEL_CALL(cuda::_LinearSearchKernel,
      nb, nt, 0, thr_entry->stream,
      csr.indptr.Ptr<IdType>(), csr.indices.Ptr<IdType>(),
      CSRHasData(csr)? csr.data.Ptr<IdType>() : nullptr,
      rows.Ptr<IdType>(), cols.Ptr<IdType>(),
      row_stride, col_stride, rstlen,
      return_eids ? nullptr : weights.Ptr<DType>(), filler, rst.Ptr<DType>());
  return rst;
}

template NDArray CSRGetData<kDLGPU, int32_t, float>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, float filler);
template NDArray CSRGetData<kDLGPU, int64_t, float>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, float filler);
template NDArray CSRGetData<kDLGPU, int32_t, double>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, double filler);
template NDArray CSRGetData<kDLGPU, int64_t, double>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, double filler);

// For CSRGetData<XPU, IdType>(CSRMatrix, NDArray, NDArray)
template NDArray CSRGetData<kDLGPU, int32_t, int32_t>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, int32_t filler);
template NDArray CSRGetData<kDLGPU, int64_t, int64_t>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, int64_t filler);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
