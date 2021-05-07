/*!
 *  Copyright (c) 2020 by Contributors
 * \file aten/cpu/sddmm.cc
 * \brief SDDMM C APIs and definitions.
 */
#include "sddmm_binary_ops.h"
#include "./sddmm.h"
#include <dgl/array.h>

namespace dgl {
namespace aten {

#define SWITCH_RHS(rhs_target, RhsTarget, ...)             \
  do {                                                     \
    if ((rhs_target) == 0) {                               \
      constexpr int RhsTarget = 0;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 1) {                        \
      constexpr int RhsTarget = 1;                         \
      { __VA_ARGS__ }                                      \
    } else if ((rhs_target) == 2) {                        \
      constexpr int RhsTarget = 2;                         \
      { __VA_ARGS__ }                                      \
    } else {                                               \
      LOG(INFO) << "Invalid rhs target: " << (rhs_target); \
    }                                                      \
  } while (0)

#define SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, ...) \
  do {                                                                   \
    if ((lhs_target) == 0) {                                             \
      constexpr int LhsTarget = 0;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 1) {                                      \
      constexpr int LhsTarget = 1;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else if ((lhs_target) == 2) {                                      \
      constexpr int LhsTarget = 2;                                       \
      SWITCH_RHS(rhs_target, RhsTarget, __VA_ARGS__);                    \
    } else {                                                             \
      LOG(INFO) << "Invalid lhs target: " << (lhs_target);               \
    }                                                                    \
  } while (0)

#define SWITCH_BITS(bits, DType, ...)                           \
  do {                                                          \
    if ((bits) == 16 || (bits) == 32) {                         \
      typedef float DType;                                      \
      { __VA_ARGS__ }                                           \
    } else if ((bits) == 64) {                                  \
      typedef double DType;                                     \
      { __VA_ARGS__ }                                           \
    } else {                                                    \
      LOG(FATAL) << "Data type not recognized with bits " << bits; \
    }                                                           \
  } while (0)


/*! \brief Generalized SDDMM on Csr format. */
template <int XPU, typename IdType, int bits>
void SDDMMCsr(const std::string& sddmm_op,
              const BcastOff& bcast,
              const CSRMatrix& csr,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target) {
  SWITCH_BITS(bits, DType, {
    SWITCH_OP_SDDMM(sddmm_op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        cpu::SDDMMCsr<IdType, DType, Op, LhsTarget, RhsTarget>(bcast, csr, lhs, rhs, out);
      });
    });
  });
}

template void SDDMMCsr<kDLCPU, int32_t, 16>(
    const std::string& sddmm_op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLCPU, int64_t, 16>(
    const std::string& sddmm_op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLCPU, int32_t, 32>(
    const std::string& sddmm_op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLCPU, int64_t, 32>(
    const std::string& sddmm_op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLCPU, int32_t, 64>(
    const std::string& sddmm_op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCsr<kDLCPU, int64_t, 64>(
    const std::string& sddmm_op, const BcastOff& bcast, const CSRMatrix& csr,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);


template <int XPU, typename IdType, int bits>
void SDDMMCoo(const std::string& sddmm_op,
              const BcastOff& bcast,
              const COOMatrix& coo,
              NDArray lhs,
              NDArray rhs,
              NDArray out,
              int lhs_target,
              int rhs_target) {
  SWITCH_BITS(bits, DType, {
    SWITCH_OP_SDDMM(sddmm_op, Op, {
      SWITCH_TARGET(lhs_target, rhs_target, LhsTarget, RhsTarget, {
        cpu::SDDMMCoo<IdType, DType, Op, LhsTarget, RhsTarget>(bcast, coo, lhs, rhs, out);
      });
    });
  });
}

template void SDDMMCoo<kDLCPU, int32_t, 16>(
    const std::string& sddmm_op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLCPU, int64_t, 16>(
    const std::string& sddmm_op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLCPU, int32_t, 32>(
    const std::string& sddmm_op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLCPU, int64_t, 32>(
    const std::string& sddmm_op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLCPU, int32_t, 64>(
    const std::string& sddmm_op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);
template void SDDMMCoo<kDLCPU, int64_t, 64>(
    const std::string& sddmm_op, const BcastOff& bcast, const COOMatrix& coo,
    NDArray lhs, NDArray rhs, NDArray out,
    int lhs_target, int rhs_target);


}  // namespace aten
}  // namespace dgl
