#ifndef DGL_KERNEL_BINARY_REDUCE_COMMON_H_
#define DGL_KERNEL_BINARY_REDUCE_COMMON_H_

#include <string>
#include "./common.h"

namespace dgl {
namespace kernel {
namespace binary_op {

static const std::string kReduceSum = "sum";
static const std::string kReduceMax = "max";
static const std::string kReduceMin = "min";
static const std::string kReduceMean = "mean";
static const std::string kReduceProd = "prod";
static const std::string kReduceNone = "none";

enum Target {
  kSrc = 0,
  kDst,
  kEdge,
};
}  // namespace binary_op


// functor for no-op
template <typename Ret, typename ... Args>
struct Nop {
  static DGLDEVICE DGLINLINE Ret Call(Args ... args) {
    return 0;
  }
};

// Select src
struct SelectSrc {
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return src;
  }
};

// Select dst
struct SelectDst {
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return dst;
  }
};

// Select edge
struct SelectEdge {
  template <typename T>
  static DGLDEVICE DGLINLINE T Call(T src, T edge, T dst) {
    return edge;
  }
};

#define TARGET_SWITCH(val, Selector, ...)        \
  switch (val) {                                 \
    case dgl::kernel::binary_op::kSrc:           \
      {                                          \
      typedef SelectSrc Selector;                \
      {__VA_ARGS__}                              \
      }                                          \
      break;                                     \
    case dgl::kernel::binary_op::kDst:           \
      {                                          \
      typedef SelectDst Selector;                \
      {__VA_ARGS__}                              \
      }                                          \
      break;                                     \
    case dgl::kernel::binary_op::kEdge:          \
      {                                          \
      typedef SelectEdge Selector;               \
      {__VA_ARGS__}                              \
      }                                          \
      break;                                     \
    default:                                     \
      LOG(FATAL) << "Invalid selector: " << val; \
  };

#define GEN_TARGET(GEN, ...) \
  GEN(__VA_ARGS__, SelectSrc, SelectDst, SelectDst) \
  GEN(__VA_ARGS__, SelectSrc, SelectDst, SelectEdge) \
  GEN(__VA_ARGS__, SelectDst, SelectSrc, SelectDst) \
  GEN(__VA_ARGS__, SelectDst, SelectSrc, SelectEdge) \
  GEN(__VA_ARGS__, SelectSrc, SelectEdge, SelectDst) \
  GEN(__VA_ARGS__, SelectSrc, SelectEdge, SelectEdge) \
  GEN(__VA_ARGS__, SelectEdge, SelectSrc, SelectDst) \
  GEN(__VA_ARGS__, SelectEdge, SelectSrc, SelectEdge) \
  GEN(__VA_ARGS__, SelectEdge, SelectDst, SelectDst) \
  GEN(__VA_ARGS__, SelectEdge, SelectDst, SelectEdge) \
  GEN(__VA_ARGS__, SelectDst, SelectEdge, SelectDst) \
  GEN(__VA_ARGS__, SelectDst, SelectEdge, SelectEdge)

// direct id
template <int XPU, typename IdxType>
struct DirectId {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType id, IdxType* shuffle_ids) {
    return id;
  }
};

// id mapped by another array
template <int XPU, typename IdxType>
struct IndirectId {
  static DGLDEVICE DGLINLINE IdxType Call(IdxType id, IdxType* shuffle_ids);
};

#define MAPPING_SWITCH(val, XPU, MapType, ...) \
  if (val->ndim == 0) {                        \
    typedef DirectId<int64_t> MapType;         \
    {__VA_ARGS__}                              \
  } else {                                     \
    typedef IndirectId<XPU, int64_t> MapType;  \
    {__VA_ARGS__}                              \
  }

// common binary functors
template <typename DType>
struct BinaryAdd {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs + rhs;
  }
};

template <typename DType>
struct BinaryMul {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs * rhs;
  }
};

template <typename DType>
struct BinarySub {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs - rhs;
  }
};

template <typename DType>
struct BinaryDiv {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs / rhs;
  }
};

template <typename DType>
struct BinaryUseLhs {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return lhs;
  }
};

template <typename DType>
struct BinaryUseRhs {
  static DGLDEVICE DGLINLINE DType Call(DType lhs, DType rhs) {
    return rhs;
  }
};

#define BINARY_OP_SWITCH(val, DType, OpType, ...)   \
  if (val == "add") {                               \
    typedef BinaryAdd<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "sub") {                        \
    typedef BinarySub<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "mul") {                        \
    typedef BinaryMul<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else if (val == "div") {                        \
    typedef BinaryDiv<DType> OpType;                \
    {__VA_ARGS__}                                   \
  } else {                                          \
    LOG(FATAL) << "Unsupported binary op: " << val; \
  }

#define GEN_BINARY_OP(GEN, ...) \
  GEN(__VA_ARGS__, BinaryAdd) \
  GEN(__VA_ARGS__, BinarySub) \
  GEN(__VA_ARGS__, BinaryMul) \
  GEN(__VA_ARGS__, BinaryDiv)

// functors for reducers
template <int XPU, typename DType>
struct ReduceSum { };

template <int XPU, typename DType>
struct ReduceMax { };

template <int XPU, typename DType>
struct ReduceMin { };

template <int XPU, typename DType>
struct ReduceMean { };

template <int XPU, typename DType>
struct ReduceProd { };

template <int XPU, typename DType>
struct ReduceNone { };

#define REDUCER_SWITCH(val, XPU, DType, RedType, ...)   \
  if (val == binary_op::kReduceSum) {              \
    typedef ReduceSum<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceMax) {       \
    typedef ReduceMax<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceMin) {       \
    typedef ReduceMin<XPU, DType> RedType;         \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceMean) {      \
    typedef ReduceMean<XPU, DType> RedType;        \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceProd) {      \
    typedef ReduceProd<XPU, DType> RedType;        \
    {__VA_ARGS__}                                  \
  } else if (val == binary_op::kReduceNone) {      \
    typedef ReduceNone<XPU, DType> RedType;        \
    {__VA_ARGS__}                                  \
  } else {                                         \
    LOG(FATAL) << "Unsupported reducer: " << val;  \
  }

}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_BINARY_REDUCE_COMMON_H_
