/*!
 *  Copyright (c) 2020 by Contributors
 * \file kernel/cpu/segment_reduce.cc
 * \brief Segment reduce C APIs and definitions.
 */
#include "./segment_reduce.h"
#include <dgl/array.h>
#include <string>
#include "./spmm_binary_ops.h"

namespace dgl {
namespace aten {

/*! \brief Segment Reduce operator. */
template <int XPU, typename IdType, int bits>
void SegmentReduce(
    const std::string& op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg) {
  if (op == "sum") {
    SWITCH_BITS(bits, DType, {
      cpu::SegmentSum<IdType, DType>(feat, offsets, out);
    });
  } else if (op == "max" || op == "min") {
    if (op == "max") {
      SWITCH_BITS(bits, DType, {
        cpu::SegmentCmp<IdType, DType, cpu::op::Max<DType>>(
            feat, offsets, out, arg);
      });
    } else {
      SWITCH_BITS(bits, DType, {
          cpu::SegmentCmp<IdType, DType, cpu::op::Min<DType>>(
              feat, offsets, out, arg);
      });
    }
  } else {
    LOG(FATAL) << "Unsupported reduce function " << op;
  }
}

/* \brief Segment Gemm operator. */
template <int XPU, int bits>
void SegmentGemm(
    NDArray A, NDArray B, NDArray C,
    NDArray m, NDArray n, NDArray k,
    bool transA, bool transB) {
  LOG(FATAL) << "Not implemented yet";
}

/*! \brief Scatter Add.*/
template <int XPU, typename IdType, int bits>
void ScatterAdd(NDArray feat,
                NDArray idx,
                NDArray out) {
  SWITCH_BITS(bits, DType, {
    cpu::ScatterAdd<IdType, DType>(feat, idx, out);
  });
}

/*! \brief Backward function of segment cmp.*/
template <int XPU, typename IdType, int bits>
void BackwardSegmentCmp(
    NDArray feat,
    NDArray arg,
    NDArray out) {
  SWITCH_BITS(bits, DType, {
    cpu::BackwardSegmentCmp<IdType, DType>(feat, arg, out);
  });
}

template void SegmentGemm<kDLCPU, 16>(
    NDArray A, NDArray B, NDArray C,
    NDArray m, NDArray n, NDArray k,
    bool transA, bool transB);
template void SegmentGemm<kDLCPU, 32>(
    NDArray A, NDArray B, NDArray C,
    NDArray m, NDArray n, NDArray k,
    bool transA, bool transB);
template void SegmentGemm<kDLCPU, 64>(
    NDArray A, NDArray B, NDArray C,
    NDArray m, NDArray n, NDArray k,
    bool transA, bool transB);
template void SegmentReduce<kDLCPU, int32_t, 16>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLCPU, int64_t, 16>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLCPU, int32_t, 32>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLCPU, int64_t, 32>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLCPU, int32_t, 64>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void SegmentReduce<kDLCPU, int64_t, 64>(
    const std::string &op,
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg);
template void ScatterAdd<kDLCPU, int32_t, 16>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLCPU, int64_t, 16>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLCPU, int32_t, 32>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLCPU, int64_t, 32>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLCPU, int32_t, 64>(
    NDArray feat,
    NDArray idx,
    NDArray out);
template void ScatterAdd<kDLCPU, int64_t, 64>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int32_t, 16>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int64_t, 16>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int32_t, 32>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int64_t, 32>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int32_t, 64>(
    NDArray feat,
    NDArray arg,
    NDArray out);
template void BackwardSegmentCmp<kDLCPU, int64_t, 64>(
    NDArray feat,
    NDArray arg,
    NDArray out);

}  // namespace aten
}  // namespace dgl
