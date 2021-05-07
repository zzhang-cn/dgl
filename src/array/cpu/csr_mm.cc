/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/cpu/csr_mm.cc
 * \brief CSR Matrix Multiplication
 */

#include <dgl/array.h>
#include <parallel_hashmap/phmap.h>
#include <vector>
#include "array_utils.h"

namespace dgl {

using dgl::runtime::NDArray;

namespace aten {

namespace {

// TODO(BarclayII): avoid using map for sorted CSRs
template <typename IdType>
void CountNNZPerRow(
    const IdType* A_indptr,
    const IdType* A_indices,
    const IdType* B_indptr,
    const IdType* B_indices,
    IdType* C_indptr_data,
    int64_t M) {
  phmap::flat_hash_set<IdType> set;
#pragma omp parallel for firstprivate(set)
  for (int64_t i = 0; i < M; ++i) {
    set.clear();
    for (IdType u = A_indptr[i]; u < A_indptr[i + 1]; ++u) {
      IdType w = A_indices[u];
      for (IdType v = B_indptr[w]; v < B_indptr[w + 1]; ++v)
        set.insert(B_indices[v]);
    }
    C_indptr_data[i] = set.size();
  }
}

template <typename IdType>
int64_t ComputeIndptrInPlace(IdType* C_indptr_data, int64_t M) {
  int64_t nnz = 0;
  IdType len = 0;
  for (IdType i = 0; i < M; ++i) {
    len = C_indptr_data[i];
    C_indptr_data[i] = nnz;
    nnz += len;
  }
  C_indptr_data[M] = nnz;
  return nnz;
}

template <typename IdType, typename DType>
void ComputeIndicesAndData(
    const IdType* A_indptr,
    const IdType* A_indices,
    const IdType* A_eids,
    const DType* A_data,
    const IdType* B_indptr,
    const IdType* B_indices,
    const IdType* B_eids,
    const DType* B_data,
    const IdType* C_indptr_data,
    IdType* C_indices_data,
    DType* C_weights_data,
    int64_t M) {
  phmap::flat_hash_map<IdType, DType> map;
#pragma omp parallel for firstprivate(map)
  for (int64_t i = 0; i < M; ++i) {
    map.clear();
    for (IdType u = A_indptr[i]; u < A_indptr[i + 1]; ++u) {
      IdType w = A_indices[u];
      DType vA = A_data[A_eids ? A_eids[u] : u];
      for (IdType v = B_indptr[w]; v < B_indptr[w + 1]; ++v) {
        IdType t = B_indices[v];
        DType vB = B_data[B_eids ? B_eids[v] : v];
        map[t] += vA * vB;
      }
    }

    IdType v = C_indptr_data[i];
    for (auto it : map) {
      C_indices_data[v] = it.first;
      C_weights_data[v] = it.second;
      ++v;
    }
  }
}

};  // namespace

template <int XPU, typename IdType, typename DType>
std::pair<CSRMatrix, NDArray> CSRMM(
    const CSRMatrix& A,
    NDArray A_weights,
    const CSRMatrix& B,
    NDArray B_weights) {
  CHECK_EQ(A.num_cols, B.num_rows) << "A's number of columns must equal to B's number of rows";
  const bool A_has_eid = !IsNullArray(A.data);
  const bool B_has_eid = !IsNullArray(B.data);
  const IdType* A_indptr = A.indptr.Ptr<IdType>();
  const IdType* A_indices = A.indices.Ptr<IdType>();
  const IdType* A_eids = A_has_eid ? A.data.Ptr<IdType>() : nullptr;
  const IdType* B_indptr = B.indptr.Ptr<IdType>();
  const IdType* B_indices = B.indices.Ptr<IdType>();
  const IdType* B_eids = B_has_eid ? B.data.Ptr<IdType>() : nullptr;
  const DType* A_data = A_weights.Ptr<DType>();
  const DType* B_data = B_weights.Ptr<DType>();
  const int64_t M = A.num_rows;
  const int64_t P = B.num_cols;

  IdArray C_indptr = IdArray::Empty({M + 1}, A.indptr->dtype, A.indptr->ctx);
  IdType* C_indptr_data = C_indptr.Ptr<IdType>();

  CountNNZPerRow<IdType>(A_indptr, A_indices, B_indptr, B_indices, C_indptr_data, M);
  int64_t nnz = ComputeIndptrInPlace<IdType>(C_indptr_data, M);
  // Allocate indices and weights array
  IdArray C_indices = IdArray::Empty({nnz}, A.indices->dtype, A.indices->ctx);
  NDArray C_weights = NDArray::Empty({nnz}, A_weights->dtype, A_weights->ctx);
  IdType* C_indices_data = C_indices.Ptr<IdType>();
  DType* C_weights_data = C_weights.Ptr<DType>();

  ComputeIndicesAndData<IdType, DType>(
      A_indptr, A_indices, A_eids, A_data,
      B_indptr, B_indices, B_eids, B_data,
      C_indptr_data, C_indices_data, C_weights_data, M);

  return {
      CSRMatrix(M, P, C_indptr, C_indices, NullArray(C_indptr->dtype, C_indptr->ctx)),
      C_weights};
}

template std::pair<CSRMatrix, NDArray> CSRMM<kDLCPU, int32_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLCPU, int64_t, float>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLCPU, int32_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);
template std::pair<CSRMatrix, NDArray> CSRMM<kDLCPU, int64_t, double>(
    const CSRMatrix&, NDArray, const CSRMatrix&, NDArray);

};  // namespace aten
};  // namespace dgl
