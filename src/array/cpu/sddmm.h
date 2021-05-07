/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/sddmm.h
 * \brief SDDMM CPU kernel function header.
 */
#ifndef DGL_ARRAY_CPU_SDDMM_H_
#define DGL_ARRAY_CPU_SDDMM_H_

#include <dgl/array.h>
#include <dgl/bcast.h>
#include <limits>
#include <memory>
#include "../selector.h"
#include "sddmm_binary_ops.h"
#if !defined(_WIN32)
#include "intel/cpu_support.h"
#endif

namespace dgl {
namespace aten {
namespace cpu {
/*!
 * \brief CPU kernel of g-SDDMM on Csr format.
 * \param bcast Broadcast information.
 * \param csr The Csr matrix.
 * \param lhs The left hand side operand feature.
 * \param rhs The right hand size operand feature.
 * \param out The result feature on edges.
 * \note it uses node parallel strategy, different threads are responsible
 *       for the computation of different nodes.
 */
template <typename IdType, typename DType, typename Op, int LhsTarget = 0,
          int RhsTarget = 2>
void SDDMMCsr(const BcastOff& bcast, const CSRMatrix& csr, NDArray lhs,
              NDArray rhs, NDArray out) {
  const bool has_idx = !IsNullArray(csr.data);
  const IdType* indptr = csr.indptr.Ptr<IdType>();
  const IdType* indices = csr.indices.Ptr<IdType>();
  const IdType* edges = csr.data.Ptr<IdType>();
  const DType* X = lhs.Ptr<DType>();
  const DType* Y = rhs.Ptr<DType>();
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;
  DType* O = out.Ptr<DType>();
#pragma omp parallel for
  for (IdType rid = 0; rid < csr.num_rows; ++rid) {
    const IdType row_start = indptr[rid], row_end = indptr[rid + 1];
    for (IdType j = row_start; j < row_end; ++j) {
      const IdType cid = indices[j];
      const IdType eid = has_idx ? edges[j] : j;
      DType* out_off = O + eid * dim;
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
        const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
        const DType* lhs_off =
          Op::use_lhs ? X + Selector<LhsTarget>::Call(rid, eid, cid) * lhs_dim +
                          lhs_add * reduce_size
                      : nullptr;
        const DType* rhs_off =
          Op::use_rhs ? Y + Selector<RhsTarget>::Call(rid, eid, cid) * rhs_dim +
                          rhs_add * reduce_size
                      : nullptr;
        out_off[k] = Op::Call(lhs_off, rhs_off, reduce_size);
      }
    }
  }
}

/*!
 * \brief CPU kernel of g-SDDMM on Coo format.
 * \param bcast Broadcast information.
 * \param coo The COO matrix.
 * \param lhs The left hand side operand feature.
 * \param rhs The right hand size operand feature.
 * \param out The result feature on edges.
 * \note it uses edge parallel strategy, different threads are responsible
 *       for the computation of different edges.
 */
template <typename IdType, typename DType, typename Op, int LhsTarget = 0,
          int RhsTarget = 2>
void SDDMMCoo(const BcastOff& bcast, const COOMatrix& coo, NDArray lhs,
              NDArray rhs, NDArray out) {
  const bool has_idx = !IsNullArray(coo.data);
  const IdType* row = coo.row.Ptr<IdType>();
  const IdType* col = coo.col.Ptr<IdType>();
  const IdType* edges = coo.data.Ptr<IdType>();
  const DType* X = lhs.Ptr<DType>();
  const DType* Y = rhs.Ptr<DType>();
  const int64_t dim = bcast.out_len, lhs_dim = bcast.lhs_len,
                rhs_dim = bcast.rhs_len, reduce_size = bcast.reduce_size;
  DType* O = out.Ptr<DType>();
  const int64_t nnz = coo.row->shape[0];
#if !defined(_WIN32)
#ifdef USE_AVX
  typedef dgl::ElemWiseUpdate<Op> ElemWise;
  /* Prepare an assembler kernel */
  static std::unique_ptr<ElemWise> asm_kernel_ptr(
    (dgl::IntelKernel<>::IsEnabled()) ? new ElemWise() : nullptr);
  /* Distribute the kernel among OMP threads */
  ElemWise* cpu_spec = (asm_kernel_ptr && asm_kernel_ptr->applicable())
                         ? asm_kernel_ptr.get()
                         : nullptr;
  if (cpu_spec && dim > 8 && !bcast.use_bcast) {
#pragma omp parallel for
    for (IdType i = 0; i < nnz; ++i) {
      const IdType rid = row[i];
      const IdType cid = col[i];
      const IdType eid = has_idx ? edges[i] : i;
      DType* out_off = O + eid * dim;
      const DType* lhs_off =
        Op::use_lhs ? X + Selector<LhsTarget>::Call(rid, eid, cid) * lhs_dim
                    : nullptr;
      const DType* rhs_off =
        Op::use_rhs ? Y + Selector<RhsTarget>::Call(rid, eid, cid) * rhs_dim
                    : nullptr;
      cpu_spec->run(out_off, lhs_off, rhs_off, dim, reduce_size);
    }
  } else {
#endif  // USE_AVX
#endif  // _WIN32

#pragma omp parallel for
    for (IdType i = 0; i < nnz; ++i) {
      const IdType rid = row[i];
      const IdType cid = col[i];
      const IdType eid = has_idx ? edges[i] : i;
      DType* out_off = O + eid * dim;
      for (int64_t k = 0; k < dim; ++k) {
        const int64_t lhs_add = bcast.use_bcast ? bcast.lhs_offset[k] : k;
        const int64_t rhs_add = bcast.use_bcast ? bcast.rhs_offset[k] : k;
        const DType* lhs_off =
          Op::use_lhs ? X + Selector<LhsTarget>::Call(rid, eid, cid) * lhs_dim +
                          lhs_add * reduce_size
                      : nullptr;
        const DType* rhs_off =
          Op::use_rhs ? Y + Selector<RhsTarget>::Call(rid, eid, cid) * rhs_dim +
                          rhs_add * reduce_size
                      : nullptr;
        out_off[k] = Op::Call(lhs_off, rhs_off, reduce_size);
      }
    }
#if !defined(_WIN32)
#ifdef USE_AVX
  }
#endif  // USE_AVX
#endif  // _WIN32
}

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SDDMM_H_
