/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/csr_sort.cc
 * \brief CSR sorting
 */
#include <dgl/array.h>
#include <numeric>
#include <algorithm>
#include <vector>

namespace dgl {
namespace aten {
namespace impl {

///////////////////////////// CSRIsSorted /////////////////////////////
template <DLDeviceType XPU, typename IdType>
bool CSRIsSorted(CSRMatrix csr) {
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  bool ret = true;

  for (int64_t row = 0; row < csr.num_rows; ++row) {
    if (!ret)
      continue;
    for (IdType i = indptr[row] + 1; i < indptr[row + 1]; ++i) {
      if (indices[i - 1] > indices[i]) {
        ret = false;
        break;
      }
    }
  }
  return ret;
}

template bool CSRIsSorted<kDLCPU, int64_t>(CSRMatrix csr);
template bool CSRIsSorted<kDLCPU, int32_t>(CSRMatrix csr);

///////////////////////////// CSRSort /////////////////////////////

template <DLDeviceType XPU, typename IdType>
void CSRSort_(CSRMatrix* csr) {
  typedef std::pair<IdType, IdType> ShufflePair;
  const int64_t num_rows = csr->num_rows;
  const int64_t nnz = csr->indices->shape[0];
  const IdType* indptr_data = static_cast<IdType*>(csr->indptr->data);
  IdType* indices_data = static_cast<IdType*>(csr->indices->data);
  if (!CSRHasData(*csr)) {
    csr->data = aten::Range(0, nnz, csr->indptr->dtype.bits, csr->indptr->ctx);
  }
  IdType* eid_data = static_cast<IdType*>(csr->data->data);
#pragma omp parallel
  {
    std::vector<ShufflePair> reorder_vec;
#pragma omp for
    for (int64_t row = 0; row < num_rows; row++) {
      const int64_t num_cols = indptr_data[row + 1] - indptr_data[row];
      IdType *col = indices_data + indptr_data[row];
      IdType *eid = eid_data + indptr_data[row];

      reorder_vec.resize(num_cols);
      for (int64_t i = 0; i < num_cols; i++) {
        reorder_vec[i].first = col[i];
        reorder_vec[i].second = eid[i];
      }
      std::sort(reorder_vec.begin(), reorder_vec.end(),
                [](const ShufflePair &e1, const ShufflePair &e2) {
                  return e1.first < e2.first;
                });
      for (int64_t i = 0; i < num_cols; i++) {
        col[i] = reorder_vec[i].first;
        eid[i] = reorder_vec[i].second;
      }
    }
  }
  csr->sorted = true;
}

template void CSRSort_<kDLCPU, int64_t>(CSRMatrix* csr);
template void CSRSort_<kDLCPU, int32_t>(CSRMatrix* csr);

template <DLDeviceType XPU, typename IdType, typename TagType>
NDArray CSRSortByTag(const CSRMatrix* csr, const IdArray tag_array, int64_t num_tags, CSRMatrix* output) {
std::cerr << "Called" << std::endl;
  const auto indptr_data = static_cast<const IdType *>(csr->indptr->data);
  const auto indices_data = static_cast<const IdType *>(csr->indices->data);
  const auto eid_array = aten::CSRHasData(*csr) ? csr->data :
    aten::Range(0, csr->indices->shape[0], csr->indptr->dtype.bits, csr->indptr->ctx);
  const auto eid_data = static_cast<const IdType *>(csr->data->data);
  const auto tag_data = static_cast<const TagType *>(tag_array->data);
  const int64_t num_rows = csr->num_rows;

  NDArray tag_pos = NDArray::Empty({csr->num_rows, num_tags + 1},
      csr->indptr->dtype, csr->indptr->ctx);
  auto tag_pos_data = static_cast<IdType *>(tag_pos->data);
  std::fill(tag_pos_data, tag_pos_data + csr->num_rows * (num_tags + 1), 0);

  auto out_indptr_data = static_cast<IdType *>(output->indptr->data);
  auto out_indices_data = static_cast<IdType *>(output->indices->data);
  auto out_eid_data = static_cast<IdType *>(output->data->data);

// #pragma omp parallel for
  for (IdType src = 0 ; src < num_rows ; ++src) {
    const IdType start = indptr_data[src];
    const IdType end = indptr_data[src + 1];

    auto tag_pos_row = tag_pos_data + src * (num_tags + 1);
    std::vector<IdType> pointer(num_tags, 0);

    for (IdType ptr = start ; ptr < end ; ++ptr) {
      const IdType dst = indices_data[ptr];
      const TagType tag = tag_data[dst];
      CHECK_LT(tag, num_tags);
      ++tag_pos_row[tag + 1];
    } // count

    for (TagType tag = 1 ; tag <= num_tags; ++tag) {
      tag_pos_row[tag] += tag_pos_row[tag - 1];
    } // cumulate

    for (IdType ptr = start ; ptr < end ; ++ptr) {
      IdType dst = indices_data[ptr];
      IdType eid = eid_data[ptr];
      TagType tag = tag_data[dst];
      IdType offset = tag_pos_row[tag] + pointer[tag];
      CHECK_LT(offset, tag_pos_row[tag + 1]);
      ++pointer[tag];

      out_indices_data[start + offset] = dst;
      out_eid_data[start + offset] = eid;
    }
  }

  output->sorted = false;
  return tag_pos;
}

template NDArray CSRSortByTag<kDLCPU, int64_t, int64_t>(
    const CSRMatrix* csr, const IdArray tag, int64_t num_tags, CSRMatrix* output);
template NDArray CSRSortByTag<kDLCPU, int64_t, int32_t>(
    const CSRMatrix* csr, const IdArray tag, int64_t num_tags, CSRMatrix* output);
template NDArray CSRSortByTag<kDLCPU, int32_t, int64_t>(
    const CSRMatrix* csr, const IdArray tag, int64_t num_tags, CSRMatrix* output);
template NDArray CSRSortByTag<kDLCPU, int32_t, int32_t>(
    const CSRMatrix* csr, const IdArray tag, int64_t num_tags, CSRMatrix* output);

}  // namespace impl
}  // namespace aten
}  // namespace dgl
