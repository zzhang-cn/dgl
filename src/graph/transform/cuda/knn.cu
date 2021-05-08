/*!
 *  Copyright (c) 2020 by Contributors
 * \file graph/transform/cuda/knn.cu
 * \brief k-nearest-neighbor (KNN) implementation (cuda)
 */

#include <dgl/array.h>
#include <dgl/runtime/device_api.h>
#include <algorithm>
#include <string>
#include <vector>
#include <limits>
#include "../../../runtime/cuda/cuda_common.h"
#include "../../../array/cuda/utils.h"
#include "../knn.h"

namespace dgl {
namespace transform {
namespace impl {
/*!
 * \brief Utility class used to avoid linker errors with extern
 *  unsized shared memory arrays with templated type
 */
template <typename Type>
struct SharedMemory {
  __device__ inline operator Type* () {
    extern __shared__ int __smem[];
    return reinterpret_cast<Type*>(__smem);
  }

  __device__ inline operator const Type* () const {
    extern __shared__ int __smem[];
    return reinterpret_cast<Type*>(__smem);
  }
};

// specialize for double to avoid unaligned memory
// access compile errors
template <>
struct SharedMemory<double> {
  __device__ inline operator double* () {
    extern __shared__ double __smem_d[];
    return reinterpret_cast<double*>(__smem_d);
  }

  __device__ inline operator const double* () const {
    extern __shared__ double __smem_d[];
    return reinterpret_cast<double*>(__smem_d);
  }
};

/*!
 * \brief Brute force kNN kernel. Compute distance for each pair of input points and get
 *  the result directly (without a distance matrix).
 */
template <typename FloatType, typename IdType>
__global__ void bruteforce_knn_kernel(const FloatType* data_points, const IdType* data_offsets,
                                      const FloatType* query_points, const IdType* query_offsets,
                                      const int k, FloatType* dists, IdType* query_out,
                                      IdType* data_out, const int num_batches,
                                      const int feature_size) {
  const int64_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
  int64_t batch_idx = 0;
  for (IdType b = 0; b < num_batches + 1; ++b) {
    if (query_offsets[b] > q_idx) { batch_idx = b - 1; break; }
  }
  const IdType data_start = data_offsets[batch_idx], data_end = data_offsets[batch_idx + 1];

  for (IdType k_idx = 0; k_idx < k; ++k_idx) {
    query_out[q_idx * k + k_idx] = q_idx;
    dists[q_idx * k + k_idx] = std::numeric_limits<FloatType>::max();
  }
  FloatType worst_dist = std::numeric_limits<FloatType>::max();

  for (IdType d_idx = data_start; d_idx < data_end; ++d_idx) {
    FloatType tmp_dist = 0;
    IdType dim_idx = 0;
    bool early_stop = false;

    // expand loop (x4), #pragma unroll has poor performance here
    for (; dim_idx < feature_size - 3; dim_idx += 4) {
      FloatType diff0 = query_points[q_idx * feature_size + dim_idx]
      - data_points[d_idx * feature_size + dim_idx];
      FloatType diff1 = query_points[q_idx * feature_size + dim_idx + 1]
      - data_points[d_idx * feature_size + dim_idx + 1];
      FloatType diff2 = query_points[q_idx * feature_size + dim_idx + 2]
      - data_points[d_idx * feature_size + dim_idx + 2];
      FloatType diff3 = query_points[q_idx * feature_size + dim_idx + 3]
      - data_points[d_idx * feature_size + dim_idx + 3];
      tmp_dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;

      // stop if current distance > all top-k distances.
      if (tmp_dist > worst_dist) {
        early_stop = true;
        dim_idx = feature_size;
        break;
      }
    }

    // last few elements
    for (; dim_idx < feature_size; ++dim_idx) {
      FloatType diff = query_points[q_idx * feature_size + dim_idx]
        - data_points[d_idx * feature_size + dim_idx];
      tmp_dist += diff * diff;

      if (tmp_dist > worst_dist) {
        early_stop = true;
        break;
      }
    }

    if (early_stop) continue;

    // maintain a monotonic array by "insert sort"
    IdType out_offset = q_idx * k;
    for (IdType k1 = 0; k1 < k; ++k1) {
      if (dists[out_offset + k1] > tmp_dist) {
        for (IdType k2 = k - 1; k2 > k1; --k2) {
          dists[out_offset + k2] = dists[out_offset + k2 - 1];
          data_out[out_offset + k2] = data_out[out_offset + k2 - 1];
        }
        dists[out_offset + k1] = tmp_dist;
        data_out[out_offset + k1] = d_idx;
        worst_dist = dists[out_offset + k - 1];
        break;
      }
    }
  }
}

/*!
 * \brief Same as bruteforce_knn_kernel, but use shared memory as buffer.
 *  This kernel divides query points and data points into blocks. For each
 *  query block, it will make a loop over all data blocks and compute distances.
 *  This kernel is faster when the dimension of input points is not large.
 */
template <typename FloatType, typename IdType>
__global__ void bruteforce_knn_share_kernel(const FloatType* data_points,
                                            const IdType* data_offsets,
                                            const FloatType* query_points,
                                            const IdType* query_offsets,
                                            const IdType* block_batch_id,
                                            const IdType* local_block_id,
                                            const int k, FloatType* dists,
                                            IdType* query_out, IdType* data_out,
                                            const int num_batches,
                                            const int feature_size) {
  const IdType block_idx = static_cast<IdType>(blockIdx.x);
  const IdType block_size = static_cast<IdType>(blockDim.x);
  const IdType batch_idx = block_batch_id[block_idx];
  const IdType local_bid = local_block_id[block_idx];
  const IdType query_start = query_offsets[batch_idx] + block_size * local_bid;
  const IdType query_end = min(query_start + block_size, query_offsets[batch_idx + 1]);
  const IdType query_idx = query_start + threadIdx.x;
  const IdType data_start = data_offsets[batch_idx];
  const IdType data_end = data_offsets[batch_idx + 1];

  // shared memory: points in block + distance buffer + result buffer
  FloatType* data_buff = SharedMemory<FloatType>();
  FloatType* query_buff = data_buff + block_size * feature_size;
  FloatType* dist_buff = query_buff + block_size * feature_size;
  IdType* res_buff = reinterpret_cast<IdType*>(dist_buff + block_size * k);
  FloatType worst_dist = std::numeric_limits<FloatType>::max();

  // initialize dist buff with inf value
  for (auto i = 0; i < k; ++i) {
    dist_buff[threadIdx.x * k + i] = std::numeric_limits<FloatType>::max();
  }

  // load query data to shared memory
  if (query_idx < query_end) {
    for (auto i = 0; i < feature_size; ++i) {
      // to avoid bank conflict, we use transpose here
      query_buff[threadIdx.x + i * block_size] = query_points[query_idx * feature_size + i];
    }
  }

  // perform computation on each tile
  for (auto tile_start = data_start; tile_start < data_end; tile_start += block_size) {
    // each thread load one data point into the shared memory
    IdType load_idx = tile_start + threadIdx.x;
    if (load_idx < data_end) {
      for (auto i = 0; i < feature_size; ++i) {
        data_buff[threadIdx.x * feature_size + i] = data_points[load_idx * feature_size + i];
      }
    }
    __syncthreads();

    // compute distance for one tile
    IdType true_block_size = min(data_end - tile_start, block_size);
    if (query_idx < query_end) {
      for (IdType d_idx = 0; d_idx < true_block_size; ++d_idx) {
        FloatType tmp_dist = 0;
        bool early_stop = false;
        IdType dim_idx = 0;

        for (; dim_idx < feature_size - 3; dim_idx += 4) {
          FloatType diff0 = query_buff[threadIdx.x + block_size * (dim_idx)]
            - data_buff[d_idx * feature_size + dim_idx];
          FloatType diff1 = query_buff[threadIdx.x + block_size * (dim_idx + 1)]
            - data_buff[d_idx * feature_size + dim_idx + 1];
          FloatType diff2 = query_buff[threadIdx.x + block_size * (dim_idx + 2)]
            - data_buff[d_idx * feature_size + dim_idx + 2];
          FloatType diff3 = query_buff[threadIdx.x + block_size * (dim_idx + 3)]
            - data_buff[d_idx * feature_size + dim_idx + 3];

          tmp_dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;

          if (tmp_dist > worst_dist) {
            early_stop = true;
            dim_idx = feature_size;
            break;
          }
        }

        for (; dim_idx < feature_size; ++dim_idx) {
          const FloatType diff = query_buff[threadIdx.x + dim_idx * block_size]
            - data_buff[d_idx * feature_size + dim_idx];
          tmp_dist += diff * diff;

          if (tmp_dist > worst_dist) {
            early_stop = true;
            break;
          }
        }

        if (early_stop) continue;

        for (IdType k1 = 0; k1 < k; ++k1) {
          if (dist_buff[threadIdx.x * k + k1] > tmp_dist) {
            for (IdType k2 = k - 1; k2 > k1; --k2) {
              dist_buff[threadIdx.x * k + k2] = dist_buff[threadIdx.x * k + k2 - 1];
              res_buff[threadIdx.x * k + k2] = res_buff[threadIdx.x * k + k2 - 1];
            }
            dist_buff[threadIdx.x * k + k1] = tmp_dist;
            res_buff[threadIdx.x * k + k1] = d_idx + tile_start;
            worst_dist = dist_buff[threadIdx.x * k + k - 1];
            break;
          }
        }
      }
    }
  }

  // copy result to global memory
  if (query_idx < query_end) {
    for (auto i = 0; i < k; ++i) {
      dists[query_idx * k + i] = dist_buff[threadIdx.x * k + i];
      data_out[query_idx * k + i] = res_buff[threadIdx.x * k + i];
      query_out[query_idx * k + i] = query_idx;
    }
  }
}

/*!
 * \brief Brute force kNN. Compute distance for each pair of input points and get
 *  the result directly (without a distance matrix).
 *
 * \tparam FloatType The type of input points.
 * \tparam IdType The type of id.
 * \param data_points NDArray of dataset points.
 * \param data_offsets offsets of point index in data points.
 * \param query_points NDArray of query points
 * \param query_offsets offsets of point index in query points.
 * \param k the number of nearest points
 * \param result output array
 */
template <typename FloatType, typename IdType>
void BruteForceKNNCuda(const NDArray& data_points, const IdArray& data_offsets,
                       const NDArray& query_points, const IdArray& query_offsets,
                       const int k, IdArray result) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const auto& ctx = data_points->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* data_points_data = data_points.Ptr<FloatType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];

  FloatType* dists = static_cast<FloatType*>(device->AllocWorkspace(
    ctx, k * query_points->shape[0] * sizeof(FloatType)));

  const int64_t block_size = cuda::FindNumThreads(query_points->shape[0]);
  const int64_t num_blocks = (query_points->shape[0] - 1) / block_size + 1;
  CUDA_KERNEL_CALL(bruteforce_knn_kernel, num_blocks, block_size, 0, thr_entry->stream,
    data_points_data, data_offsets_data, query_points_data, query_offsets_data,
    k, dists, query_out, data_out, batch_size, feature_size);

  device->FreeWorkspace(ctx, dists);
}

/*!
 * \brief Brute force kNN with shared memory.
 *  This function divides query points and data points into blocks. For each
 *  query block, it will make a loop over all data blocks and compute distances.
 *  It will be faster when the dimension of input points is not large.
 *
 * \tparam FloatType The type of input points.
 * \tparam IdType The type of id.
 * \param data_points NDArray of dataset points.
 * \param data_offsets offsets of point index in data points.
 * \param query_points NDArray of query points
 * \param query_offsets offsets of point index in query points.
 * \param k the number of nearest points
 * \param result output array
 */
template <typename FloatType, typename IdType>
void BruteForceKNNSharedCuda(const NDArray& data_points, const IdArray& data_offsets,
                             const NDArray& query_points, const IdArray& query_offsets,
                             const int k, IdArray result) {
  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  const auto& ctx = data_points->ctx;
  auto device = runtime::DeviceAPI::Get(ctx);
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* data_points_data = data_points.Ptr<FloatType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];

  // get max shared memory per block in bytes
  int max_sharedmem_per_block = 0;
  CUDA_CALL(cudaDeviceGetAttribute(
    &max_sharedmem_per_block, cudaDevAttrMaxSharedMemoryPerBlock, ctx.device_id));

  // determine block size and number of blocks according to share memory size per block
  // This could be improved by using GPU implementation in future
  ::std::vector<IdType> data_offsets_host(data_offsets->shape[0]);
  device->CopyDataFromTo(data_offsets_data, 0, data_offsets_host.data(),
                         0, data_offsets->shape[0] * sizeof(IdType),
                         ctx, DLContext{kDLCPU, 0}, data_offsets->dtype,
                         nullptr);
  int64_t single_shared_mem = (k + 2 * feature_size) * sizeof(FloatType) + k * sizeof(IdType);
  const int64_t block_size = cuda::FindNumThreads(max_sharedmem_per_block / single_shared_mem);
  int64_t num_blocks = 0;

#pragma omp parallel for reduction(+:num_blocks)
  for (auto i = 0; i < batch_size; ++i) {
    IdType curr_batch_length = data_offsets_host[i + 1] - data_offsets_host[i];
    num_blocks += (curr_batch_length - 1) / block_size + 1;
  }

  // for each block, find its batch id
  ::std::vector<IdType> block_batch_id_host(num_blocks), local_block_id_host(num_blocks);
  int64_t prefix_index = 0;
  for (auto i = 0; i < batch_size; ++i) {
    IdType curr_batch_length = data_offsets_host[i + 1] - data_offsets_host[i];
    int64_t batch_blocks = (curr_batch_length - 1) / block_size + 1;

#pragma omp parallel for
    for (auto j = 0; j < batch_blocks; ++j) {
      block_batch_id_host[j + prefix_index] = static_cast<IdType>(i);
      local_block_id_host[j + prefix_index] = static_cast<IdType>(j);
    }
    prefix_index += batch_blocks;
  }

  IdType* block_batch_id = static_cast<IdType*>(device->AllocWorkspace(
    ctx, num_blocks * sizeof(IdType)));
  IdType* local_block_id = static_cast<IdType*>(device->AllocWorkspace(
    ctx, num_blocks * sizeof(IdType)));
  FloatType* dists = static_cast<FloatType*>(device->AllocWorkspace(
    ctx, k * query_points->shape[0] * sizeof(FloatType)));
  device->CopyDataFromTo(
    block_batch_id_host.data(), 0, block_batch_id, 0,
    num_blocks * sizeof(IdType), DLContext{kDLCPU, 0},
    ctx, data_offsets->dtype, thr_entry->stream);
  device->CopyDataFromTo(
    local_block_id_host.data(), 0, local_block_id, 0,
    num_blocks * sizeof(IdType), DLContext{kDLCPU, 0},
    ctx, data_offsets->dtype, thr_entry->stream);

  CUDA_KERNEL_CALL(bruteforce_knn_share_kernel, num_blocks, block_size,
    single_shared_mem * block_size, thr_entry->stream, data_points_data,
    data_offsets_data, query_points_data, query_offsets_data,
    block_batch_id, local_block_id, k, dists, query_out,
    data_out, batch_size, feature_size);

  device->FreeWorkspace(ctx, dists);
  device->FreeWorkspace(ctx, local_block_id);
  device->FreeWorkspace(ctx, block_batch_id);
}
}  // namespace impl

template <DLDeviceType XPU, typename FloatType, typename IdType>
void KNN(const NDArray& data_points, const IdArray& data_offsets,
         const NDArray& query_points, const IdArray& query_offsets,
         const int k, IdArray result, const std::string& algorithm) {
  if (algorithm == std::string("bruteforce")) {
    impl::BruteForceKNNCuda<FloatType, IdType>(
      data_points, data_offsets, query_points, query_offsets, k, result);
  } else if (algorithm == std::string("bruteforce-sharemem")) {
    impl::BruteForceKNNSharedCuda<FloatType, IdType>(
      data_points, data_offsets, query_points, query_offsets, k, result);
  } else {
    LOG(FATAL) << "Algorithm " << algorithm << " is not supported on CUDA.";
  }
}

template void KNN<kDLGPU, float, int32_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLGPU, float, int64_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLGPU, double, int32_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLGPU, double, int64_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);

}  // namespace transform
}  // namespace dgl
