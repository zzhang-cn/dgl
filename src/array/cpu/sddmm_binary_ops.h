/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cpu/sddmm_binary_ops.h
 * \brief SDDMM CPU Binary ops.
 */
#ifndef DGL_ARRAY_CPU_SDDMM_BINARY_OPS_H_
#define DGL_ARRAY_CPU_SDDMM_BINARY_OPS_H_
#include <dgl/array.h>
#include <dgl/bcast.h>
#include <limits>

namespace dgl {
namespace aten {
namespace cpu {
namespace sddmm_op {

//////////////////////////////// binary operators on CPU
///////////////////////////////////
template <typename DType>
struct Add {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off,
                           int64_t len = 1) {
    return *lhs_off + *rhs_off;
  }
};
template <typename DType>
constexpr bool Add<DType>::use_lhs;
template <typename DType>
constexpr bool Add<DType>::use_rhs;

template <typename DType>
struct Sub {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off,
                           int64_t len = 1) {
    return *lhs_off - *rhs_off;
  }
};
template <typename DType>
constexpr bool Sub<DType>::use_lhs;
template <typename DType>
constexpr bool Sub<DType>::use_rhs;

template <typename DType>
struct Mul {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off,
                           int64_t len = 1) {
    return *lhs_off * *rhs_off;
  }
};
template <typename DType>
constexpr bool Mul<DType>::use_lhs;
template <typename DType>
constexpr bool Mul<DType>::use_rhs;

template <typename DType>
struct Div {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off,
                           int64_t len = 1) {
    return *lhs_off / *rhs_off;
  }
};
template <typename DType>
constexpr bool Div<DType>::use_lhs;
template <typename DType>
constexpr bool Div<DType>::use_rhs;

template <typename DType>
struct CopyLhs {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = false;
  inline static DType Call(const DType* lhs_off, const DType*,
                           int64_t len = 1) {
    return *lhs_off;
  }
};
template <typename DType>
constexpr bool CopyLhs<DType>::use_lhs;
template <typename DType>
constexpr bool CopyLhs<DType>::use_rhs;

template <typename DType>
struct CopyRhs {
  typedef DType type;
  static constexpr bool use_lhs = false;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType*, const DType* rhs_off,
                           int64_t len = 1) {
    return *rhs_off;
  }
};
template <typename DType>
constexpr bool CopyRhs<DType>::use_lhs;
template <typename DType>
constexpr bool CopyRhs<DType>::use_rhs;

template <typename DType>
struct Dot {
  typedef DType type;
  static constexpr bool use_lhs = true;
  static constexpr bool use_rhs = true;
  inline static DType Call(const DType* lhs_off, const DType* rhs_off,
                           int64_t len = 1) {
    DType rst = 0;
    for (int64_t l = 0; l < len; ++l) {
      rst += lhs_off[l] * rhs_off[l];
      // std::cout << "rst: " << rst << std::endl;
    }
    return rst;
  }
};
template <typename DType>
constexpr bool Dot<DType>::use_lhs;
template <typename DType>
constexpr bool Dot<DType>::use_rhs;

#define SWITCH_OP_SDDMM(sddmm_op, Op, ...)                             \
  do {                                                                 \
    if ((sddmm_op) == "add") {                                         \
      typedef dgl::aten::cpu::sddmm_op::Add<DType> Op;                 \
      { __VA_ARGS__ }                                                  \
    } else if ((sddmm_op) == "sub") {                                  \
      typedef dgl::aten::cpu::sddmm_op::Sub<DType> Op;                 \
      { __VA_ARGS__ }                                                  \
    } else if ((sddmm_op) == "mul") {                                  \
      typedef dgl::aten::cpu::sddmm_op::Mul<DType> Op;                 \
      { __VA_ARGS__ }                                                  \
    } else if ((sddmm_op) == "div") {                                  \
      typedef dgl::aten::cpu::sddmm_op::Div<DType> Op;                 \
      { __VA_ARGS__ }                                                  \
    } else if ((sddmm_op) == "copy_lhs") {                             \
      typedef dgl::aten::cpu::sddmm_op::CopyLhs<DType> Op;             \
      { __VA_ARGS__ }                                                  \
    } else if ((sddmm_op) == "copy_rhs") {                             \
      typedef dgl::aten::cpu::sddmm_op::CopyRhs<DType> Op;             \
      { __VA_ARGS__ }                                                  \
    } else if ((sddmm_op) == "dot") {                                  \
      typedef dgl::aten::cpu::sddmm_op::Dot<DType> Op;                 \
      { __VA_ARGS__ }                                                  \
    } else {                                                           \
      LOG(FATAL) << "Unsupported SDDMM binary operator: " << sddmm_op; \
    }                                                                  \
  } while (0)

}  // namespace sddmm_op

}  // namespace cpu
}  // namespace aten
}  // namespace dgl

#endif  // DGL_ARRAY_CPU_SDDMM_BINARY_OPS_H_
