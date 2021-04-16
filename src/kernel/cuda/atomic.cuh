/*!
 *  Copyright (c) 2019 by Contributors
 * \file kernel/cuda/atomic.cuh
 * \brief Atomic functions
 */
#ifndef DGL_KERNEL_CUDA_ATOMIC_H_
#define DGL_KERNEL_CUDA_ATOMIC_H_

#include <cuda_runtime.h>
#if __CUDA_ARCH__ >= 600
#include <cuda_fp16.h>
#endif

namespace dgl {
namespace kernel {
namespace cuda {

// Type trait for selecting code type
template <int Bytes> struct Code { };

template <> struct Code<4> {
  typedef unsigned int Type;
};

template <> struct Code<8> {
  typedef unsigned long long int Type;
};

// Helper class for converting to/from atomicCAS compatible types.
template <typename T> struct Cast {
  typedef typename Code<sizeof(T)>::Type Type;
  static __device__ __forceinline__ Type Encode(T val) {
    return static_cast<Type>(val);
  }
  static __device__ __forceinline__ T Decode(Type code) {
    return static_cast<T>(code);
  }
};

template <> struct Cast<float> {
  typedef Code<sizeof(float)>::Type Type;
  static __device__ __forceinline__ Type Encode(float val) {
    return __float_as_uint(val);
  }
  static __device__ __forceinline__ float Decode(Type code) {
    return __uint_as_float(code);
  }
};

template <> struct Cast<double> {
  typedef Code<sizeof(double)>::Type Type;
  static __device__ __forceinline__ Type Encode(double val) {
    return __double_as_longlong(val);
  }
  static __device__ __forceinline__ double Decode(Type code) {
    return __longlong_as_double(code);
  }
};

#define DEFINE_ATOMIC(NAME) \
  template <typename T>                                          \
  __device__ __forceinline__ T Atomic##NAME(T* addr, T val) {    \
    typedef typename Cast<T>::Type CT;                           \
    CT* addr_as_ui = reinterpret_cast<CT*>(addr);                \
    CT old = *addr_as_ui;                                        \
    CT assumed = old;                                            \
    do {                                                         \
      assumed = old;                                             \
      old = atomicCAS(addr_as_ui, assumed,                       \
          Cast<T>::Encode(OP(val, Cast<T>::Decode(old))));       \
    } while (assumed != old);                                    \
    return Cast<T>::Decode(old);                                 \
  }

#define OP(a, b) max(a, b)
DEFINE_ATOMIC(Max)
#undef OP

#define OP(a, b) min(a, b)
DEFINE_ATOMIC(Min)
#undef OP

#define OP(a, b) a + b
DEFINE_ATOMIC(Add)
#undef OP

#if __CUDA_ARCH__ >= 200
template <>
__device__ __forceinline__ float AtomicAdd<float>(float* addr, float val) {
  return atomicAdd(addr, val);
}
#endif  // __CUDA_ARCH__

#if __CUDA_ARCH__ >= 600
template <>
__device__ __forceinline__ double AtomicAdd<double>(double* addr, double val) {
  return atomicAdd(addr, val);
}
#endif

#if defined(CUDART_VERSION) && CUDART_VERSION >= 10000
#if __CUDA_ARCH__ >= 600
template <>
__device__ __forceinline__ __half2 AtomicAdd<__half2>(__half2* addr, __half2 val) {
  return atomicAdd(addr, val);
}
#endif  // __CUDA_ARCH__

#if __CUDA_ARCH__ >= 700
template <>
__device__ __forceinline__ __half AtomicAdd<__half>(__half* addr, __half val) {
  return atomicAdd(addr, val);
}
#endif  // __CUDA_ARCH__
#endif

#define OP(a, b) a * b
DEFINE_ATOMIC(Mul)
#undef OP

/**
* \brief Performs an atomic compare-and-swap on 64 bit integers. That is,
* it the word `old` at the memory location `address`, computes
* `(old == compare ? val : old)` , and stores the result back to memory at
* the same address.
*
* \param address The address to perform the atomic operation on.
* \param compare The value to compare to.
* \param val The new value to conditionally store.
*
* \return The old value at the address.
*/
inline __device__ int64_t AtomicCAS(
    int64_t * const address,
    const int64_t compare,
    const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type*>(address),
                   static_cast<Type>(compare),
                   static_cast<Type>(val));
}

/**
* \brief Performs an atomic compare-and-swap on 32 bit integers. That is,
* it the word `old` at the memory location `address`, computes
* `(old == compare ? val : old)` , and stores the result back to memory at
* the same address.
*
* \param address The address to perform the atomic operation on.
* \param compare The value to compare to.
* \param val The new value to conditionally store.
*
* \return The old value at the address.
*/
inline __device__ int32_t AtomicCAS(
    int32_t * const address,
    const int32_t compare,
    const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicCAS(reinterpret_cast<Type*>(address),
                   static_cast<Type>(compare),
                   static_cast<Type>(val));
}

inline __device__ int64_t AtomicMax(
    int64_t * const address,
    const int64_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = unsigned long long int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type*>(address),
                   static_cast<Type>(val));
}

inline __device__ int32_t AtomicMax(
    int32_t * const address,
    const int32_t val) {
  // match the type of "::atomicCAS", so ignore lint warning
  using Type = int; // NOLINT

  static_assert(sizeof(Type) == sizeof(*address), "Type width must match");

  return atomicMax(reinterpret_cast<Type*>(address),
                   static_cast<Type>(val));
}

}  // namespace cuda
}  // namespace kernel
}  // namespace dgl

#endif  // DGL_KERNEL_CUDA_ATOMIC_H_
