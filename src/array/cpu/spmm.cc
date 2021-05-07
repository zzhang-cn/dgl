/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cpu/spmm.cc
 * \brief SPMM C APIs and definitions.
 */
#include "./spmm.h"
#include <dgl/array.h>

namespace dgl {
namespace aten {

/*! \brief Generalized SpMM on Csr format. */
template <int XPU, typename IdType, int bits>
void SpMMCsr(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const CSRMatrix& csr,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cpu::SpMMSumCsr<IdType, DType, Op>(bcast, csr, ufeat, efeat, out);
      });
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        if (reduce == "max")
          cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Max<DType>>(
              bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
        else
          cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Min<DType>>(
              bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

/*! \brief Generalized SpMM on Csr format with Heterograph support. */
template <int XPU, typename IdType, int bits>
void SpMMCsrHetero(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const std::vector<CSRMatrix>& vec_csr,
             std::vector<NDArray> vec_ufeat,
             NDArray efeat,
             std::vector<NDArray> vec_out,
             std::vector<NDArray> out_aux,
             const std::vector<dgl_type_t> ufeat_eid,
             const std::vector<dgl_type_t> out_eid) {
  if (reduce == "sum") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        /* Call  SpMM for each relation type */
        for (dgl_type_t etype = 0; etype < ufeat_eid.size(); ++etype) {
          const dgl_type_t src_id = ufeat_eid[etype];
          const dgl_type_t dst_id = out_eid[etype];
          CSRMatrix csr = vec_csr[etype];
          NDArray ufeat = vec_ufeat[src_id];
          NDArray out = vec_out[dst_id];
          cpu::SpMMSumCsr<IdType, DType, Op>(bcast, csr, ufeat, efeat, out);
        }
      });
    });
  } 
  // else if (reduce == "max" || reduce == "min") {
  //   SWITCH_BITS(bits, DType, {
  //     SWITCH_OP(op, Op, {
  //       if (reduce == "max")
  //         cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Max<DType>>(
  //             bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
  //       else
  //         cpu::SpMMCmpCsr<IdType, DType, Op, cpu::op::Min<DType>>(
  //             bcast, csr, ufeat, efeat, out, out_aux[0], out_aux[1]);
  //     });
  //   });
  // } 
  else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

template void SpMMCsr<kDLCPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCsr<kDLCPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);


template void SpMMCsrHetero<kDLCPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    std::vector<NDArray> ufeat, NDArray efeat, std::vector<NDArray> out, 
    std::vector<NDArray> out_aux, std::vector<dgl_type_t> ufeat_eid,
    std::vector<dgl_type_t> out_eid);
template void SpMMCsrHetero<kDLCPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    std::vector<NDArray> ufeat, NDArray efeat, std::vector<NDArray> out, 
    std::vector<NDArray> out_aux, std::vector<dgl_type_t> ufeat_eid,
    std::vector<dgl_type_t> out_eid);
template void SpMMCsrHetero<kDLCPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    std::vector<NDArray> ufeat, NDArray efeat, std::vector<NDArray> out, 
    std::vector<NDArray> out_aux, std::vector<dgl_type_t> ufeat_eid,
    std::vector<dgl_type_t> out_eid);
template void SpMMCsrHetero<kDLCPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    std::vector<NDArray> ufeat, NDArray efeat, std::vector<NDArray> out, 
    std::vector<NDArray> out_aux, std::vector<dgl_type_t> ufeat_eid,
    std::vector<dgl_type_t> out_eid);
template void SpMMCsrHetero<kDLCPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    std::vector<NDArray> ufeat, NDArray efeat, std::vector<NDArray> out, 
    std::vector<NDArray> out_aux, std::vector<dgl_type_t> ufeat_eid,
    std::vector<dgl_type_t> out_eid);
template void SpMMCsrHetero<kDLCPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const std::vector<CSRMatrix>& csr,
    std::vector<NDArray> ufeat, NDArray efeat, std::vector<NDArray> out, 
    std::vector<NDArray> out_aux, std::vector<dgl_type_t> ufeat_eid,
    std::vector<dgl_type_t> out_eid);



/*! \brief Generalized SpMM on Coo format. */
template <int XPU, typename IdType, int bits>
void SpMMCoo(const std::string& op, const std::string& reduce,
             const BcastOff& bcast,
             const COOMatrix& coo,
             NDArray ufeat,
             NDArray efeat,
             NDArray out,
             std::vector<NDArray> out_aux) {
  if (reduce == "sum") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        cpu::SpMMSumCoo<IdType, DType, Op>(bcast, coo, ufeat, efeat, out);
      });
    });
  } else if (reduce == "max" || reduce == "min") {
    SWITCH_BITS(bits, DType, {
      SWITCH_OP(op, Op, {
        if (reduce == "max")
          cpu::SpMMCmpCoo<IdType, DType, Op, cpu::op::Max<DType>>(
              bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
        else
          cpu::SpMMCmpCoo<IdType, DType, Op, cpu::op::Min<DType>>(
              bcast, coo, ufeat, efeat, out, out_aux[0], out_aux[1]);
      });
    });
  } else {
    LOG(FATAL) << "Unsupported SpMM reducer: " << reduce;
  }
}

template void SpMMCoo<kDLCPU, int32_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int64_t, 16>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int32_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int64_t, 32>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int32_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);
template void SpMMCoo<kDLCPU, int64_t, 64>(
    const std::string& op, const std::string& reduce,
    const BcastOff& bcast, const COOMatrix& coo,
    NDArray ufeat, NDArray efeat, NDArray out, std::vector<NDArray> out_aux);


}  // namespace aten
}  // namespace dgl
