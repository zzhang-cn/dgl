/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cpu/csr_get_data.cc
 * \brief Retrieve entries of a CSR matrix
 */
#include <dgl/array.h>
#include <vector>
#include <unordered_set>
#include <numeric>
#include "array_utils.h"

namespace dgl {

using runtime::NDArray;

namespace aten {
namespace impl {

template <DLDeviceType XPU, typename IdType>
void CollectDataFromSorted(const IdType *indices_data, const IdType *data,
                           const IdType start, const IdType end, const IdType col,
                           std::vector<IdType> *ret_vec) {
  const IdType *start_ptr = indices_data + start;
  const IdType *end_ptr = indices_data + end;
  auto it = std::lower_bound(start_ptr, end_ptr, col);
  // This might be a multi-graph. We need to collect all of the matched
  // columns.
  for (; it != end_ptr; it++) {
    // If the col exist
    if (*it == col) {
      IdType idx = it - indices_data;
      ret_vec->push_back(data? data[idx] : idx);
    } else {
      // If we find a column that is different, we can stop searching now.
      break;
    }
  }
}

template <DLDeviceType XPU, typename IdType, typename DType>
NDArray CSRGetData(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, DType filler) {
  const int64_t rowlen = rows->shape[0];
  const int64_t collen = cols->shape[0];

  CHECK((rowlen == collen) || (rowlen == 1) || (collen == 1))
    << "Invalid row and col id array.";

  const int64_t row_stride = (rowlen == 1 && collen != 1) ? 0 : 1;
  const int64_t col_stride = (collen == 1 && rowlen != 1) ? 0 : 1;
  const IdType* row_data = static_cast<IdType*>(rows->data);
  const IdType* col_data = static_cast<IdType*>(cols->data);

  const IdType* indptr_data = static_cast<IdType*>(csr.indptr->data);
  const IdType* indices_data = static_cast<IdType*>(csr.indices->data);
  const IdType* data = CSRHasData(csr)? static_cast<IdType*>(csr.data->data) : nullptr;

  const int64_t retlen = std::max(rowlen, collen);
  const DType* weight_data = return_eids ? nullptr : weights.Ptr<DType>();
  if (return_eids)
    BUG_IF_FAIL(DLDataTypeTraits<DType>::dtype == rows->dtype) <<
      "DType does not match row's dtype.";

  NDArray ret = Full(filler, retlen, rows->ctx);
  DType* ret_data = ret.Ptr<DType>();

  // NOTE: In most cases, the input csr is already sorted. If not, we might need to
  //   consider sorting it especially when the number of (row, col) pairs is large.
  //   Need more benchmarks to justify the choice.

  if (csr.sorted) {
    // use binary search on each row
#pragma omp parallel for
    for (int64_t p = 0; p < retlen; ++p) {
      const IdType row_id = row_data[p * row_stride], col_id = col_data[p * col_stride];
      CHECK(row_id >= 0 && row_id < csr.num_rows) << "Invalid row index: " << row_id;
      CHECK(col_id >= 0 && col_id < csr.num_cols) << "Invalid col index: " << col_id;
      const IdType *start_ptr = indices_data + indptr_data[row_id];
      const IdType *end_ptr = indices_data + indptr_data[row_id + 1];
      auto it = std::lower_bound(start_ptr, end_ptr, col_id);
      if (it != end_ptr && *it == col_id) {
        const IdType idx = it - indices_data;
        IdType eid = data ? data[idx] : idx;
        ret_data[p] = return_eids ? eid : weight_data[eid];
      }
    }
  } else {
    // linear search on each row
#pragma omp parallel for
    for (int64_t p = 0; p < retlen; ++p) {
      const IdType row_id = row_data[p * row_stride], col_id = col_data[p * col_stride];
      CHECK(row_id >= 0 && row_id < csr.num_rows) << "Invalid row index: " << row_id;
      CHECK(col_id >= 0 && col_id < csr.num_cols) << "Invalid col index: " << col_id;
      for (IdType idx = indptr_data[row_id]; idx < indptr_data[row_id + 1]; ++idx) {
        if (indices_data[idx] == col_id) {
          IdType eid = data ? data[idx] : idx;
          ret_data[p] = return_eids ? eid : weight_data[eid];
          break;
        }
      }
    }
  }
  return ret;
}

template NDArray CSRGetData<kDLCPU, int32_t, float>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, float filler);
template NDArray CSRGetData<kDLCPU, int64_t, float>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, float filler);
template NDArray CSRGetData<kDLCPU, int32_t, double>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, double filler);
template NDArray CSRGetData<kDLCPU, int64_t, double>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, double filler);

// For CSRGetData<XPU, IdType>(CSRMatrix, NDArray, NDArray)
template NDArray CSRGetData<kDLCPU, int32_t, int32_t>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, int32_t filler);
template NDArray CSRGetData<kDLCPU, int64_t, int64_t>(
    CSRMatrix csr, NDArray rows, NDArray cols, bool return_eids, NDArray weights, int64_t filler);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
