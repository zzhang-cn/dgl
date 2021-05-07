/*!
 *  Copyright (c) 2020 by Contributors
 * \file array/cuda/segment_reduce.cuh
 * \brief Segment reduce kernel function header.
 */
#ifndef DGL_ARRAY_SEGMENT_REDUCE_CUH_
#define DGL_ARRAY_SEGMENT_REDUCE_CUH_

#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"
#include "./atomic.cuh"
#include <algorithm>
#include <vector>

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

/*!
 * \brief CUDA kernel of segment reduce.
 * \note each blockthread is responsible for aggregation on a row
 *       in the result tensor.
 */
template <typename IdType, typename DType,
          typename ReduceOp>
__global__ void SegmentReduceKernel(
    const DType* feat, const IdType* offsets,
    DType* out, IdType* arg,
    int64_t n, int64_t dim){
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      DType local_accum = ReduceOp::zero();
      IdType local_arg = -1;
      for (IdType i = offsets[row]; i < offsets[row + 1]; ++i) {
        ReduceOp::Call(&local_accum, &local_arg, feat[i * dim + col], i);
      }
      out[row * dim + col] = local_accum;
      if (ReduceOp::require_arg)
        arg[row * dim + col] = local_arg;
      col += gridDim.y * blockDim.x;
    }
  }
}

/*!
 * \brief CUDA kernel of scatter add.
 * \note each blockthread is responsible for adding a row in feature tensor
 *       to a target row in output tensor.
 */
template <typename IdType, typename DType>
__global__ void ScatterAddKernel(
    const DType *feat, const IdType *idx, DType *out,
    int64_t n, int64_t dim) {
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    const int write_row = idx[row];
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      cuda::AtomicAdd(out + write_row * dim + col, feat[row * dim + col]);
      col += gridDim.y * blockDim.x;
    }
  }
}

/*!
 * \brief CUDA kernel of backward phase in segment min/max.
 * \note each blockthread is responsible for writing a row in the
 *       result gradient tensor by lookup the ArgMin/Max for index information.
 */
template <typename IdType, typename DType>
__global__ void BackwardSegmentCmpKernel(
    const DType *feat, const IdType *arg, DType *out,
    int64_t n, int64_t dim) {
  for (int row = blockIdx.x; row < n; row += gridDim.x) {
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    while (col < dim) {
      int write_row = arg[row * dim + col];
      if (write_row >= 0) {
        out[write_row * dim + col] = feat[row * dim + col];
      }
      col += gridDim.y * blockDim.x;
    }
  }
}

/*!
 * \brief CUDA implementation of forward phase of Segment Reduce.
 * \param feat The input tensor.
 * \param offsets The offsets tensor.
 * \param out The output tensor.
 * \param arg An auxiliary tensor storing ArgMax/Min information,
 */
template <typename IdType, typename DType, typename ReduceOp>
void SegmentReduce(
    NDArray feat,
    NDArray offsets,
    NDArray out,
    NDArray arg) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* offsets_data = offsets.Ptr<IdType>();
  DType* out_data = out.Ptr<DType>();
  IdType* arg_data = arg.Ptr<IdType>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t n = out->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  // TODO(zihao): try cub's DeviceSegmentedReduce and compare the performance.
  CUDA_KERNEL_CALL((SegmentReduceKernel<IdType, DType, ReduceOp>),
      nblks, nthrs, 0, thr_entry->stream,
      feat_data, offsets_data, out_data, arg_data,
      n, dim);
}

namespace {

template <typename DType>
cublasStatus_t
Xgemm(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
      int m, int n, int k, const DType *alpha, const DType *A, int lda,
      const DType *B, int ldb, const DType *beta, DType *C, int ldc) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t
Xgemm<float> (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
              int m, int n, int k, const float *alpha, const float *A, int lda,
              const float *B, int ldb, const float *beta, float *C, int ldc) {
  return cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <>
cublasStatus_t
Xgemm<double> (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
               int m, int n, int k, const double *alpha, const double *A, int lda,
               const double *B, int ldb, const double *beta, double *C, int ldc) {
  return cublasDgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template<>
cublasStatus_t
Xgemm<half> (cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
             int m, int n, int k, const half *alpha, const half *A, int lda,
             const half *B, int ldb, const half *beta, half *C, int ldc) {
  return cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

template <typename DType>
cublasStatus_t
XgemmBatched(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const DType *alpha,
    const DType *A, int lda, int64_t stride_a,
    const DType *B, int ldb, int64_t stride_b,
    const DType *beta,
    DType *C, int ldc, int64_t stride_c,
    int batchCount) {
  LOG(INFO) << "Not supported dtype";
  return CUBLAS_STATUS_EXECUTION_FAILED;
}

template <>
cublasStatus_t
XgemmBatched<float>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const float *alpha,
    const float *A, int lda, int64_t stride_a,
    const float *B, int ldb, int64_t stride_b,
    const float *beta,
    float *C, int ldc, int64_t stride_c,
    int batch_count) {
  return cublasSgemmStridedBatched(
      handle,
      transa, transb,
      m, n, k,
      alpha,
      A, lda,
      stride_a,
      B, ldb,
      stride_b,
      beta,
      C, ldc,
      stride_c,
      batch_count
  );
}

template <>
cublasStatus_t
XgemmBatched<double>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const double *alpha,
    const double *A, int lda, int64_t stride_a,
    const double *B, int ldb, int64_t stride_b,
    const double *beta,
    double *C, int ldc, int64_t stride_c,
    int batch_count) {
  return cublasDgemmStridedBatched(
      handle,
      transa, transb,
      m, n, k,
      alpha,
      A, lda,
      stride_a,
      B, ldb,
      stride_b,
      beta,
      C, ldc,
      stride_c,
      batch_count
  );
}

template <>
cublasStatus_t
XgemmBatched<half>(
    cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb,
    int m, int n, int k,
    const half *alpha,
    const half *A, int lda, int64_t stride_a,
    const half *B, int ldb, int64_t stride_b,
    const half *beta,
    half *C, int ldc, int64_t stride_c,
    int batch_count) {
  return cublasHgemmStridedBatched(
      handle,
      transa, transb,
      m, n, k,
      alpha,
      A, lda,
      stride_a,
      B, ldb,
      stride_b,
      beta,
      C, ldc,
      stride_c,
      batch_count
  );
}

}  // namespace

#define TRANS_SWITCH(trans_x, transX, ...) do { \
  if ((trans_x)) {                              \
    constexpr bool transX = true;               \
    { __VA_ARGS__ }                             \
  } else {                                      \
    constexpr bool transX = false;              \
    { __VA_ARGS__ }                             \
  }                                             \
} while (0)

template <typename DType, bool transA, bool transB>
__global__ void SegmentGemmKernel(
    const DType* __restrict__ A,
    const DType* __restrict__ B,
    DType* __restrict__ C,
    const int64_t* __restrict__ m_arr,
    const int64_t* __restrict__ n_arr,
    const int64_t* __restrict__ k_arr,
    const int64_t* __restrict__ A_off,
    const int64_t* __restrict__ B_off,
    const int64_t* __restrict__ C_off,
    const int64_t* __restrict__ seg_indices,
    const int64_t* __restrict__ seg_offsets_m,
    const int64_t* __restrict__ seg_offsets_n) {
  // TODO(zihao): use cutlass device-wide function in the future.
  __shared__ DType A_local[32 * 32],
                   B_local[32 * 32];
  int64_t tx = threadIdx.x, ty = threadIdx.y;
  __syncthreads();
  // Preparation
  int64_t seg_id = seg_indices[blockIdx.x],
          cta_m = seg_offsets_m[blockIdx.x] * 32,
          cta_n = seg_offsets_n[blockIdx.x] * 32;
  const DType *A_global = A + A_off[seg_id],
              *B_global = B + B_off[seg_id];
  DType *C_global = C + C_off[seg_id];
  int64_t m = m_arr[seg_id],
          n = n_arr[seg_id],
          k = k_arr[seg_id];
  DType sum = 0.;
  for (int cta_k = 0; cta_k < k; cta_k += 32) {
    // Copying global data to local shared mem.
    // copy A
    A_local[ty * 32 + tx] = 0.;
    if (transA) {
      if (cta_k + ty < k && cta_m + tx < m)
        A_local[ty * 32 + tx] = A_global[(cta_k + ty) * m + (cta_m + tx)];
    } else {
      if (cta_m + ty < m && cta_k + tx < k)
        A_local[ty * 32 + tx] = A_global[(cta_m + ty) * k + (cta_k + tx)];
    }
    // copy B
    B_local[ty * 32 + tx] = 0.;
    if (transB) {
      if (cta_n + ty < n && cta_k + tx < k)
        B_local[ty * 32 + tx] = B_global[(cta_n + ty) * k + (cta_k + tx)];
    } else {
      if (cta_k + ty < k && cta_n + tx < n)
        B_local[ty * 32 + tx] = B_global[(cta_k + ty) * n + (cta_n + tx)];
    }
    __syncthreads();
#pragma unroll
    for (int iter_k = 0; iter_k < 32; iter_k++) {
      sum += (transA ? A_local[iter_k * 32 + ty] : A_local[ty * 32 + iter_k]) *\
             (transB ? B_local[tx * 32 + iter_k] : B_local[iter_k * 32 + tx]);
    }
  }
  // Write local data back to global mem.
  if (cta_m + ty < m && cta_n + tx < n)
    C_global[(cta_m + ty) * n + (cta_n + tx)] = sum;
}

/*
 * \brief CUDA implementation of forward phase of SegmentGemm.
 * \param A the concatenation of {A_i}
 * \param B the concatenation of {B_i}
 * \param C the concatenation of {C_i}
 * \param m the array of number of rows in A.
 * \param n the array of number of columns in C.
 * \param k the array of number of rows in B / number of columns in A.
 * \param trans_a if A_i need to be transposed.
 * \param trans_b if B_i need to be transposed.
 */
template <typename DType>
void SegmentGemm(NDArray A, NDArray B, NDArray C,
                 NDArray m, NDArray n, NDArray k,
                 bool trans_a, bool trans_b) {
  auto ctx = A->ctx;
  const DType *A_data = A.Ptr<DType>();
  const DType *B_data = B.Ptr<DType>();
  DType* C_data = C.Ptr<DType>();
  const int64_t *m_data = m.Ptr<int64_t>();
  const int64_t *n_data = n.Ptr<int64_t>();
  const int64_t *k_data = k.Ptr<int64_t>();
  int64_t batch_size = n->shape[0];
  auto device = runtime::DeviceAPI::Get(ctx);
  auto *thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  if (!thr_entry->cublas_handle)
    CUBLAS_CALL(cublasCreate(&(thr_entry->cublas_handle)));
  CUBLAS_CALL(cublasSetStream(thr_entry->cublas_handle, thr_entry->stream));

  std::vector<int64_t> A_off(batch_size + 1, 0),
                       B_off(batch_size + 1, 0),
                       C_off(batch_size + 1, 0);
  std::vector<int64_t> seg_indices, seg_offsets_m, seg_offsets_n;
  // collect information
  for (int i = 0; i < batch_size; ++i) {
    int64_t mi = m_data[i],
            ni = n_data[i],
            ki = k_data[i];
    A_off[i + 1] = A_off[i] + mi * ki;
    B_off[i + 1] = B_off[i] + ki * ni;
    C_off[i + 1] = C_off[i] + mi * ni;
    int seg_mi = (mi + 31) / 32,
        seg_ni = (ni + 31) / 32;
    for (int j = 0; j < seg_mi; ++j) {
      for (int k = 0; k < seg_ni; ++k) {
        seg_indices.push_back(i);
        seg_offsets_m.push_back(j);
        seg_offsets_n.push_back(k);
      }
    }
  }

  if (batch_size > 1) {
    const int nblks = seg_indices.size();
    const dim3 nthrs(32, 32);

    // allocate memory and data movement
    int64_t *pool = static_cast<int64_t*>(
        device->AllocWorkspace(ctx, (6 * batch_size + 3 * nblks) * sizeof(int64_t)));
    int64_t *m_gpu = pool,
            *n_gpu = m_gpu + batch_size,
            *k_gpu = n_gpu + batch_size,
            *A_off_gpu = k_gpu + batch_size,
            *B_off_gpu = A_off_gpu + batch_size,
            *C_off_gpu = B_off_gpu + batch_size,
            *seg_indices_gpu = C_off_gpu + batch_size,
            *seg_offsets_m_gpu = seg_indices_gpu + nblks,
            *seg_offsets_n_gpu = seg_offsets_m_gpu + nblks;
    CUDA_CALL(cudaMemcpy(m_gpu, m_data,
        batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(n_gpu, n_data,
        batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(k_gpu, k_data,
        batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(A_off_gpu, A_off.data(),
        batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(B_off_gpu, B_off.data(),
        batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(C_off_gpu, C_off.data(),
        batch_size * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(seg_indices_gpu, seg_indices.data(),
        nblks * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(seg_offsets_m_gpu, seg_offsets_m.data(),
        nblks * sizeof(int64_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(seg_offsets_n_gpu, seg_offsets_n.data(),
        nblks * sizeof(int64_t), cudaMemcpyHostToDevice));

    // trigger kernel
    TRANS_SWITCH(trans_a, transA, {
      TRANS_SWITCH(trans_b, transB, {
        CUDA_KERNEL_CALL((SegmentGemmKernel<DType, transA, transB>),
          nblks, nthrs, 0, thr_entry->stream,
          A_data, B_data, C_data,
          m_gpu, n_gpu, k_gpu,
          A_off_gpu, B_off_gpu, C_off_gpu,
          seg_indices_gpu, seg_offsets_m_gpu, seg_offsets_n_gpu
        );
      });
    });

    // deallocate memory
    device->FreeDataSpace(ctx, pool);
  } else {
    // parallelize with CUDA stream.
    for (int i = 0; i < batch_size; ++i) {
      int64_t mi = m_data[i],
              ni = n_data[i],
              ki = k_data[i];
      if (mi > 0 && ni > 0 && ki > 0) {
        DType alpha = 1., beta = 0.;
        int ldb = trans_b ? ki: ni,
            lda = trans_a ? mi: ki,
            ldc = ni;
        CUBLAS_CALL(Xgemm<DType>(
            thr_entry->cublas_handle,
            trans_b ? CUBLAS_OP_T : CUBLAS_OP_N,
            trans_a ? CUBLAS_OP_T : CUBLAS_OP_N,
            ni, mi, ki,
            &alpha,
            B_data + B_off[i], ldb,
            A_data + A_off[i], lda,
            &beta,
            C_data + C_off[i], ldc
        ));
      }
    }
  }
}

/*!
 * \brief CUDA implementation of Scatter Add (on first dimension).
 * \note math equation: out[idx[i], *] += feat[i, *]
 * \param feat The input tensor.
 * \param idx The indices tensor.
 * \param out The output tensor.
 */
template <typename IdType, typename DType>
void ScatterAdd(
    NDArray feat,
    NDArray idx,
    NDArray out) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* idx_data = idx.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();
  
  auto *thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t n = feat->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  CUDA_KERNEL_CALL((ScatterAddKernel<IdType, DType>),
                   nblks, nthrs, 0, thr_entry->stream,
                   feat_data, idx_data, out_data,
                   n, dim);
}

/*!
 * \brief CUDA implementation of backward phase of Segment Reduce with Min/Max reducer.
 * \note math equation: out[arg[i, k], k] = feat[i, k]
 * \param feat The input tensor.
 * \param arg The ArgMin/Max information, used for indexing.
 * \param out The output tensor.
 */
template <typename IdType, typename DType>
void BackwardSegmentCmp(
    NDArray feat,
    NDArray arg,
    NDArray out) {
  const DType* feat_data = feat.Ptr<DType>();
  const IdType* arg_data = arg.Ptr<IdType>();
  DType *out_data = out.Ptr<DType>();

  auto *thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  int64_t n = feat->shape[0];
  int64_t dim = 1;
  for (int i = 1; i < out->ndim; ++i)
    dim *= out->shape[i];

  const int nbx = FindNumBlocks<'x'>(n);
  const int ntx = FindNumThreads(dim);
  const int nby = FindNumBlocks<'y'>((dim + ntx - 1) / ntx);
  const int nty = 1;
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty);
  CUDA_KERNEL_CALL((BackwardSegmentCmpKernel<IdType, DType>),
                   nblks, nthrs, 0, thr_entry->stream,
                   feat_data, arg_data, out_data,
                   n, dim);
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif
