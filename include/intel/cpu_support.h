/*!
 *  Copyright (c) 2019 by Contributors
 * \file intel/cpu_support.h
 * \brief Intel CPU support
 * \author Pawel Piotrowicz <pawel.piotrowicz@intel.com>
 */
#ifndef INTEL_CPU_SUPPORT_H_
#define INTEL_CPU_SUPPORT_H_
#include <memory>
#include <tuple>
#include <type_traits>
#include "../src/array/cpu/sddmm_binary_ops.h"
#include "../src/array/cpu/spmm_binary_ops.h"
#include "dmlc/logging.h"
#include "meta_utils.h"
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace dgl {

typedef std::tuple<float, double> supported_types;

#ifndef log_intel
#define log_intel(x)                   \
  if (IntelKernel<>::IsLogEnabled()) { \
    LOG(INFO) << x;                    \
  }
#endif

static inline Xbyak::Zmm make_zmm(const Xbyak::Xmm &v) {
  return Xbyak::Zmm(v.getIdx());
}
template <int version = 0>
struct IntelKernel {
  static int64_t GetValue() {
    int64_t v = 0;
    const char *label = "DGL_CPU_INTEL_KERNEL_ENABLED";
    const char *ptr = std::getenv(label);
    if (ptr) {
      v = atoll(ptr);
      log_intel(label << "=>" << v);
    }
    return v;
  }

  static int64_t IsEnabled() {
    static int64_t r = IntelKernel<version>::GetValue();
    return r;
  }

  static int IsLogEnabled() {
    static int r = (std::getenv("DGL_CPU_INTEL_KERNEL_LOG")) ? 1 : 0;
    return r;
  }
};

/*!
 * \brief Element-wise addition kernel using Intel AVX512 instructions.
 * \note it uses AVX512,  operation [index]+=Op(..).
 */
template <class Op>
class ElemWiseAddUpdate : public Xbyak::CodeGenerator {
 public:
  typedef typename Op::type DType;
  static_assert(
    std::is_base_of<std::true_type,
                    utils::has_type<DType, supported_types>>::value,
    "Use case fail dgl::ElemWiseAddUpdate< Operator<DType> > DType is not "
    "supported !");

 protected:
  const Xbyak::Reg64 &r_out_;
  const Xbyak::Reg64 &r_left_;
  const Xbyak::Reg64 &r_right;
  const Xbyak::Reg64 &r_size_;

  /* [functional] Does kernel is applicable on this machine ? */
  bool applicable_;

 public:
  static constexpr int UNIT_SIZE_BYTES = sizeof(DType);
  static constexpr int BITS_IN_BYTES = 8;
  static constexpr int REG_BIT_SIZE = 512;
  static constexpr int UNIT_PER_REG =
    REG_BIT_SIZE / (UNIT_SIZE_BYTES * BITS_IN_BYTES);

  template <class TType, class R1, class R2,
            utils::CheckCmp<TType, float> = true>
  void alias_load(R1 r1, R2 r2) {
    vmovups(r1, r2);
  }
  template <class TType, class R1, class R2,
            utils::CheckCmp<TType, double> = true>
  void alias_load(R1 r1, R2 r2) {
    vmovupd(r1, r2);
  }

  template <class TType, class R1, class R2,
            utils::CheckCmp<TType, float> = true>
  void alias_save(R1 r1, R2 r2) {
    alias_load<TType>(r1, r2);
  }
  template <class TType, class R1, class R2,
            utils::CheckCmp<TType, double> = true>
  void alias_save(R1 r1, R2 r2) {
    alias_load<TType>(r1, r2);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_ADD(R1 r1, R2 r2, R3 r3) {
    vaddps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_ADD(R1 r1, R2 r2, R3 r3) {
    vaddpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_SUB(R1 r1, R2 r2, R3 r3) {
    vsubps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_SUB(R1 r1, R2 r2, R3 r3) {
    vsubpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_DIV(R1 r1, R2 r2, R3 r3) {
    vdivps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_DIV(R1 r1, R2 r2, R3 r3) {
    vdivpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_MUL(R1 r1, R2 r2, R3 r3) {
    vmulps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_MUL(R1 r1, R2 r2, R3 r3) {
    vmulpd(r1, r2, r3);
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::op::CopyLhs,
                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    alias_load<IType>(zmm0, ptr[r_out_ + r9 * sizeof(IType)]);
    alias_load<IType>(zmm1, ptr[r_left_ + r9 * sizeof(IType)]);
    alias_ADD<IType>(zmm2, zmm0, zmm1);
    alias_save<IType>(ptr[r_out_ + r9 * sizeof(IType)], zmm2);
  }
  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::op::CopyRhs,
                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    alias_load<IType>(zmm0, ptr[r_out_ + r9 * sizeof(IType)]);
    alias_load<IType>(zmm1, ptr[r_right + r9 * sizeof(IType)]);
    alias_ADD<IType>(zmm2, zmm0, zmm1);
    alias_save<IType>(ptr[r_out_ + r9 * sizeof(IType)], zmm2);
  }
  template <class T>
  void loop_pre() {
    alias_load<T>(zmm0, ptr[r_out_ + r9 * sizeof(T)]);
    alias_load<T>(zmm1, ptr[r_left_ + r9 * sizeof(T)]);
    alias_load<T>(zmm2, ptr[r_right + r9 * sizeof(T)]);
  }
  template <class T>
  void loop_post() {
    alias_ADD<T>(zmm2, zmm0, zmm2);
    alias_save<T>(ptr[r_out_ + r9 * sizeof(T)], zmm2);
  }
  template <class Operator, utils::Verify<Operator, ::dgl::aten::cpu::op::Add,
                                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_ADD<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }
  template <class Operator, utils::Verify<Operator, ::dgl::aten::cpu::op::Sub,
                                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_SUB<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator, utils::Verify<Operator, ::dgl::aten::cpu::op::Div,
                                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_DIV<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator, utils::Verify<Operator, ::dgl::aten::cpu::op::Mul,
                                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_MUL<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::op::CopyLhs,
                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    alias_load<IType>(make_zmm(zmm2) | mask, ptr[r_left_ + r9 * sizeof(IType)]);
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::op::CopyRhs,
                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    alias_load<IType>(make_zmm(zmm2) | mask, ptr[r_right + r9 * sizeof(IType)]);
  }

  template <class T>
  void remainder_fetch_LR(const Xbyak::Opmask mask) {
    alias_load<T>(make_zmm(zmm2) | mask, ptr[r_left_ + r9 * sizeof(T)]);
    alias_load<T>(make_zmm(zmm1) | mask, ptr[r_right + r9 * sizeof(T)]);
  }

  template <class Operator, utils::Verify<Operator, ::dgl::aten::cpu::op::Mul,
                                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_MUL<IType>(zmm2, zmm2, zmm1);
  }

  template <class Operator, utils::Verify<Operator, ::dgl::aten::cpu::op::Add,
                                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_ADD<DType>(zmm2, zmm2, zmm1);
  }

  template <class Operator, utils::Verify<Operator, ::dgl::aten::cpu::op::Div,
                                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_DIV<DType>(zmm2, zmm2, zmm1);
  }

  template <class Operator, utils::Verify<Operator, ::dgl::aten::cpu::op::Sub,
                                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_SUB<DType>(zmm2, zmm2, zmm1);
  }

  ElemWiseAddUpdate()
      : r_out_(rdi),
        r_left_(rsi),
        r_right(rdx),
        r_size_(rcx),
        applicable_(false) {
    static Xbyak::util::Cpu current_cpu;

    /* Default case for all */
    if (current_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
      /* prepare REMAINDER */
      mov(r8, r_size_);
      and_(r8,
           UNIT_PER_REG - 1);  // r8_modulo = size/(sizeof(zmm)/sizeof(float))
      xor_(r9, r9);            // reset r9
      cmp(r_size_, UNIT_PER_REG);  // if ( size < 16 ) {  }
      jl("remainder");

      /*  decrease  divident */
      sub(r_size_, r8);  // prepare alignment chunks
      cmp(r_size_, 0);   // do we have any full chunks ?
      jz("remainder");

      L("for_i");
      full_chunk_loop_operations<Op>();
      add(r9, UNIT_PER_REG);  // r9+=sizeof(zmm)/sizeof(float)
      cmp(r_size_, r9);       // more full chunks ?
      jnz("for_i");

      L("remainder");
      cmp(r8, 0);  //  do we have a remainder ?
      jz("done");
      /* prepare a bitmask for k1 */
      mov(rax, 1);
      mov(r_size_, r8);
      sal(rax, cl);
      dec(rax);        // k1= (1 << r8 )-1
      kmovw(k1, eax);  // set bitmask
      alias_load<DType>(make_zmm(zmm0) | k1,
                        ptr[r_out_ + r9 * UNIT_SIZE_BYTES]);
      remainder_operations<Op>(k1);
      alias_ADD<DType>(zmm3, zmm2, zmm0);
      alias_save<DType>(ptr[r_out_ + r9 * UNIT_SIZE_BYTES],
                        make_zmm(zmm3) | k1);
      L("done");
      applicable_ = true;
      log_intel(
        "AVX512F ElemWiseAddUpdate<Sub,Div,Mul,Add,CopyLhs,CopyRhs> cpu kernel "
        "is ready");
    }
    ret();
  }

  bool applicable() const { return applicable_; }

  template <class... P>
  void run(P... args) {
    ((void (*)(P...))(this)->getCode())(args...);
  }
};

/*!
 * \brief Element-wise kernel using Intel AVX512 instructions.
 * \note it uses AVX512,  operation [index]=Op(..).
 */
template <class Op>
class ElemWiseUpdate : public Xbyak::CodeGenerator {
 public:
  typedef typename Op::type DType;
  static_assert(
    std::is_base_of<std::true_type,
                    utils::has_type<DType, supported_types>>::value,
    "Use case fail dgl::ElemWiseUpdate< Operator<DType> > DType is not "
    "supported !");

 protected:
  const Xbyak::Reg64 &r_out_;
  const Xbyak::Reg64 &r_left_;
  const Xbyak::Reg64 &r_right;
  const Xbyak::Reg64 &r_size_;
  const Xbyak::Reg64 &r_dot_size_;

  /* [functional] Does kernel is applicable on this machine ? */
  bool applicable_;

 public:
  static constexpr int UNIT_SIZE_BYTES = sizeof(DType);
  static constexpr int BITS_IN_BYTES = 8;
  static constexpr int REG_BIT_SIZE = 512;
  static constexpr int UNIT_PER_REG =
    REG_BIT_SIZE / (UNIT_SIZE_BYTES * BITS_IN_BYTES);

  template <class TType, class R1, class R2,
            utils::CheckCmp<TType, float> = true>
  void alias_load(R1 r1, R2 r2) {
    vmovups(r1, r2);
  }
  template <class TType, class R1, class R2,
            utils::CheckCmp<TType, double> = true>
  void alias_load(R1 r1, R2 r2) {
    vmovupd(r1, r2);
  }

  template <class TType, class R1, class R2,
            utils::CheckCmp<TType, float> = true>
  void alias_save(R1 r1, R2 r2) {
    alias_load<TType>(r1, r2);
  }
  template <class TType, class R1, class R2,
            utils::CheckCmp<TType, double> = true>
  void alias_save(R1 r1, R2 r2) {
    alias_load<TType>(r1, r2);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_ADD(R1 r1, R2 r2, R3 r3) {
    vaddps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_ADD(R1 r1, R2 r2, R3 r3) {
    vaddpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_SUB(R1 r1, R2 r2, R3 r3) {
    vsubps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_SUB(R1 r1, R2 r2, R3 r3) {
    vsubpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_DIV(R1 r1, R2 r2, R3 r3) {
    vdivps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_DIV(R1 r1, R2 r2, R3 r3) {
    vdivpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_MUL(R1 r1, R2 r2, R3 r3) {
    vmulps(r1, r2, r3);
  }
  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_MUL(R1 r1, R2 r2, R3 r3) {
    vmulpd(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, float> = true>
  void alias_DOT(R1 r1, R2 r2, R3 r3) {
    vmulps(r1, r2, r3);
  }

  template <class TType, class R1, class R2, class R3,
            utils::CheckCmp<TType, double> = true>
  void alias_DOT(R1 r1, R2 r2, R3 r3) {
    vmulpd(r1, r2, r3);
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::CopyLhs,
                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    alias_load<IType>(zmm0, ptr[r_out_ + r10 * sizeof(IType)]);
    alias_load<IType>(zmm1, ptr[r_left_ + r10 * sizeof(IType)]);
    alias_ADD<IType>(zmm2, zmm0, zmm1);
    alias_save<IType>(ptr[r_out_ + r10 * sizeof(IType)], zmm2);
  }
  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::CopyRhs,
                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    alias_load<IType>(zmm0, ptr[r_out_ + r10 * sizeof(IType)]);
    alias_load<IType>(zmm1, ptr[r_right + r10 * sizeof(IType)]);
    alias_ADD<IType>(zmm2, zmm0, zmm1);
    alias_save<IType>(ptr[r_out_ + r10 * sizeof(IType)], zmm2);
  }
  template <class T>
  void loop_pre() {
    alias_load<T>(zmm0, ptr[r_out_ + r10 * sizeof(T)]);
    alias_load<T>(zmm1, ptr[r_left_ + r10 * sizeof(T)]);
    alias_load<T>(zmm2, ptr[r_right + r10 * sizeof(T)]);
  }
  template <class T>
  void loop_post() {
    alias_ADD<T>(zmm2, zmm0, zmm2);
    alias_save<T>(ptr[r_out_ + r10 * sizeof(T)], zmm2);
  }
  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::Add,
                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_ADD<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }
  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::Sub,
                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_SUB<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::Div,
                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_DIV<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::Mul,
                          supported_types> = true>
  void full_chunk_loop_operations() {
    typedef typename Operator::type IType;
    loop_pre<IType>();
    alias_MUL<IType>(zmm2, zmm1, zmm2);
    loop_post<IType>();
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::CopyLhs,
                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    alias_load<IType>(make_zmm(zmm2) | mask,
                      ptr[r_left_ + r10 * sizeof(IType)]);
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::CopyRhs,
                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    alias_load<IType>(make_zmm(zmm2) | mask,
                      ptr[r_right + r10 * sizeof(IType)]);
  }

  template <class T>
  void remainder_fetch_LR(const Xbyak::Opmask mask) {
    alias_load<T>(make_zmm(zmm2) | mask, ptr[r_left_ + r10 * sizeof(T)]);
    alias_load<T>(make_zmm(zmm1) | mask, ptr[r_right + r10 * sizeof(T)]);
    alias_load<T>(make_zmm(zmm0) | mask, ptr[r_out_ + r10 * sizeof(T)]);
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::Mul,
                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_MUL<IType>(zmm2, zmm2, zmm1);
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::Add,
                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_ADD<DType>(zmm2, zmm2, zmm1);
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::Div,
                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_DIV<DType>(zmm2, zmm2, zmm1);
  }

  template <class Operator,
            utils::Verify<Operator, ::dgl::aten::cpu::sddmm_op::Sub,
                          supported_types> = true>
  void remainder_operations(const Xbyak::Opmask mask) {
    typedef typename Operator::type IType;
    remainder_fetch_LR<IType>(mask);
    alias_SUB<DType>(zmm2, zmm2, zmm1);
  }

  ElemWiseUpdate()
      : r_out_(rdi),
        r_left_(rsi),
        r_right(rdx),
        r_size_(rcx),
        r_dot_size_(r9),
        applicable_(false) {
    static Xbyak::util::Cpu current_cpu;
    /* Default case for all */
    if (current_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
      /* prepare REMAINDER */
      mov(r8, r_size_);
      and_(r8,
           UNIT_PER_REG - 1);  // r8_modulo = size/(sizeof(zmm)/sizeof(float))
      xor_(r10, r10);          // reset r10
      cmp(r_size_, UNIT_PER_REG);  // if ( size < 16 ) {  }
      jl("remainder");

      /*  decrease  divident */
      sub(r_size_, r8);  // prepare alignment chunks
      cmp(r_size_, 0);   // do we have any full chunks ?
      jz("remainder");

      L("for_i");

      full_chunk_loop_operations<Op>();

      add(r10, UNIT_PER_REG);  // r10+=sizeof(zmm)/sizeof(float)
      cmp(r_size_, r10);       // more full chunks ?
      jnz("for_i");

      L("remainder");
      cmp(r8, 0);  //  do we have a remainder ?
      jz("done");
      /* prepare a bitmask for k1 */
      mov(rax, 1);
      mov(r_size_, r8);
      sal(rax, cl);
      dec(rax);        // k1= (1 << r8 )-1
      kmovw(k1, eax);  // set bitmask
      alias_load<DType>(make_zmm(zmm0) | k1,
                        ptr[r_out_ + r10 * UNIT_SIZE_BYTES]);
      remainder_operations<Op>(k1);
      alias_ADD<DType>(zmm2, zmm2, zmm0);
      alias_save<DType>(ptr[r_out_ + r10 * UNIT_SIZE_BYTES],
                        make_zmm(zmm2) | k1);
      L("done");
      applicable_ = true;
      log_intel("AVX512F ElemWiseUpdate<Sub,Div,Mul,Add> cpu kernel is ready");
    }
    ret();
  }

  bool applicable() const { return applicable_; }

  template <class... P>
  void run(P... args) {
    ((void (*)(P...))(this)->getCode())(args...);
  }
};

/*!
 * \brief Element-wise kernel using Intel AVX512 instructions.
 * \note it uses AVX512 , specialization for Dot<float> operation
 * [index]=Op(..).
 */
template <>
class ElemWiseUpdate<::dgl::aten::cpu::sddmm_op::Dot<float>>
    : public Xbyak::CodeGenerator {
 public:
  typedef float self_type;
  static_assert(
    std::is_base_of<std::true_type,
                    utils::has_type<self_type, supported_types>>::value,
    "Use case fail dgl::ElemWiseUpdate< Operator<DType> > float is not "
    "supported !");

 protected:
  const Xbyak::Reg64 &r_out_;
  const Xbyak::Reg64 &r_left_;
  const Xbyak::Reg64 &r_right_;
  const Xbyak::Reg64 &r_size_;
  const Xbyak::Reg64 &r_dot_size_;

  /* [functional] Does kernel is applicable on this machine ? */
  bool applicable_;

 public:
  ElemWiseUpdate()
      : r_out_(rdi),
        r_left_(rsi),
        r_right_(rdx),
        r_size_(rcx),
        r_dot_size_(r8),
        applicable_(false) {
    static Xbyak::util::Cpu current_cpu;
    /* Default case for all */
    if (current_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
      push(r10);
      push(r11);
      xor_(r10, r10);
      imul(r11, r8, 4);
      L("for_i");
      dot_product(r_left_, r_right_, r_dot_size_);  // L, R, reduce_size
      vmovss(ptr[r_out_ + r10 * 4], xmm0);
      add(r_left_, r11);
      add(r_right_, r11);
      inc(r10);
      cmp(r10, r_size_);
      jne("for_i");
      pop(r11);
      pop(r10);
      ret();

      applicable_ = true;
      log_intel("AVX512F ElemWiseUpdate<Dot<float>> cpu kernel is ready");
    }
    ret();
  }
  // rdi (left), rsi(right) , rdx(size),
  void dot_product(const Xbyak::Reg64 &reg_L, const Xbyak::Reg64 &reg_R,
                   const Xbyak::Reg64 &reg_size) {
    vxorps(xmm0, xmm0, xmm0);
    test(reg_size, reg_size);
    je("end", T_NEAR);
    xor_(rax, rax);
    cmp(reg_size, 16);
    jl("remainder");
    vxorps(zmm3, zmm3, zmm3);
    vxorps(zmm2, zmm2, zmm2);
    mov(r9, reg_size);
    and_(r9, 16 - 1);
    sub(reg_size, r9);
    L("full_chunk");
    vmovups(zmm0, ptr[reg_L + rax * 4]);
    vmovups(zmm1, ptr[reg_R + rax * 4]);
    vmulps(zmm2, zmm1, zmm0);
    vaddps(zmm3, zmm2, zmm3);
    add(rax, 16);
    cmp(reg_size, rax);
    jne("full_chunk");
    vextractf64x4(ymm0, zmm3, 0x0);
    vextractf64x4(ymm1, zmm3, 0x1);
    vaddps(ymm3, ymm1, ymm0);
    vextractf128(xmm1, ymm3, 0x0);
    vextractf128(xmm2, ymm3, 0x1);
    vaddps(xmm0, xmm1, xmm2);
    vshufps(xmm1, xmm0, xmm0, 0xb1);
    vaddps(xmm0, xmm1, xmm0);
    vshufps(xmm1, xmm0, xmm0, 0x02);
    vaddps(xmm0, xmm1, xmm0);
    cmp(r9, 0);
    je("end");
    L("set_remainder");
    add(reg_size, r9);
    L("remainder");
    vmovss(xmm1, ptr[reg_L + rax * 4]);
    vfmadd231ss(xmm0, xmm1, ptr[reg_R + rax * 4]);
    inc(rax);
    cmp(reg_size, rax);
    jne("remainder");
    L("end");
  }

  void run(self_type *out, const self_type *l, const self_type *r,
           size_t size_out, size_t reduce_size) {
    return ((void (*)(const self_type *, const self_type *, const self_type *,
                      size_t, size_t))(this)
              ->getCode())(out, l, r, size_out, reduce_size);
  }

  bool applicable() const { return applicable_; }
};

/*!
 * \brief Element-wise kernel using Intel AVX512 instructions.
 * \note it uses AVX512 , specialization for Dot<double> operation
 * [index]=Op(..).
 */
template <>
class ElemWiseUpdate<::dgl::aten::cpu::sddmm_op::Dot<double>>
    : public Xbyak::CodeGenerator {
 public:
  typedef double self_type;
  static_assert(
    std::is_base_of<std::true_type,
                    utils::has_type<self_type, supported_types>>::value,
    "Use case fail dgl::ElemWiseUpdate< Operator<DType> > double is not "
    "supported !");

 protected:
  const Xbyak::Reg64 &r_out_;
  const Xbyak::Reg64 &r_left_;
  const Xbyak::Reg64 &r_right_;
  const Xbyak::Reg64 &r_size_;
  const Xbyak::Reg64 &r_dot_size_;

  /* [functional] Does kernel is applicable on this machine ? */
  bool applicable_;

 public:
  ElemWiseUpdate()
      : r_out_(rdi),
        r_left_(rsi),
        r_right_(rdx),
        r_size_(rcx),
        r_dot_size_(r8),
        applicable_(false) {
    static Xbyak::util::Cpu current_cpu;
    /* Default case for all */
    if (current_cpu.has(Xbyak::util::Cpu::tAVX512F)) {
      push(r10);
      push(r11);
      xor_(r10, r10);
      imul(r11, r8, 8);
      L("for_i");
      dot_product(r_left_, r_right_, r_dot_size_);  // L, R, reduce_size
      vmovsd(ptr[r_out_ + r10 * 8], xmm0);
      add(r_left_, r11);
      add(r_right_, r11);
      inc(r10);
      cmp(r10, r_size_);
      jne("for_i");
      pop(r11);
      pop(r10);
      ret();

      applicable_ = true;
      log_intel("AVX512F ElemWiseUpdate<Dot<double>> cpu kernel is ready");
    }
    ret();
  }
  // rdi (left), rsi(right) , rdx(size),
  void dot_product(const Xbyak::Reg64 &reg_L, const Xbyak::Reg64 &reg_R,
                   const Xbyak::Reg64 &reg_size) {
    vxorpd(xmm0, xmm0, xmm0);
    test(reg_size, reg_size);
    je("end", T_NEAR);
    xor_(rax, rax);
    cmp(reg_size, 8);
    jl("remainder");
    vxorpd(zmm3, zmm3, zmm3);
    vxorpd(zmm2, zmm2, zmm2);
    mov(r9, reg_size);
    and_(r9, 8 - 1);
    sub(reg_size, r9);
    L("full_chunk");
    vmovupd(zmm0, ptr[reg_L + rax * 8]);
    vmovupd(zmm1, ptr[reg_R + rax * 8]);
    vmulpd(zmm2, zmm1, zmm0);
    vaddpd(zmm3, zmm2, zmm3);
    add(rax, 8);
    cmp(reg_size, rax);
    jne("full_chunk");
    vextractf64x4(ymm0, zmm3, 0x0);
    vextractf64x4(ymm1, zmm3, 0x1);
    vaddpd(ymm3, ymm1, ymm0);
    vextractf128(xmm1, ymm3, 0x0);
    vextractf128(xmm2, ymm3, 0x1);
    vaddpd(xmm0, xmm1, xmm2);
    movapd(xmm1, xmm0);
    shufpd(xmm1, xmm1, 0x1);
    vaddpd(xmm0, xmm1, xmm0);
    cmp(r9, 0);
    je("end");
    L("set_remainder");
    add(reg_size, r9);
    L("remainder");
    vmovsd(xmm1, ptr[reg_L + rax * 8]);
    vfmadd231sd(xmm0, xmm1, ptr[reg_R + rax * 8]);
    inc(rax);
    cmp(reg_size, rax);
    jne("remainder");
    L("end");
  }
  void run(self_type *out, const self_type *l, const self_type *r,
           size_t size_out, size_t reduce_size) {
    return ((void (*)(const self_type *, const self_type *, const self_type *,
                      size_t, size_t))(this)
              ->getCode())(out, l, r, size_out, reduce_size);
  }
  bool applicable() const { return applicable_; }
};

}  // namespace dgl

#endif  // INTEL_CPU_SUPPORT_H_
