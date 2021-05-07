/*!
 *  Copyright (c) 2021 by Contributors
 * \file nccl_api.cc
 * \brief Implementation of wrapper around NCCL routines.
 */


#include "nccl_api.h"

#include <dgl/array.h>
#include <dgl/aten/array_ops.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/device_api.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/registry.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cmath>
#include <sstream>
#include <iomanip>
#include <utility>
#include <vector>
#include <memory>
#include <string>
#include <limits>

#include "cuda_common.h"
#include "../../kernel/cuda/atomic.cuh"
#include "../../array/cuda/dgl_cub.cuh"
#include "../../array/cuda/array_index_select.cuh"

#define NCCL_CALL(func) \
{ \
  ncclResult_t result = func; \
  if (result != ncclSuccess) { \
      LOG(FATAL)                                                        \
          << "NCCLError: " #func " failed with error: " << result;            \
  } \
}

namespace dgl {

using namespace kernel::cuda;

namespace runtime {
namespace cuda {

namespace {

enum class AllToAllMode : int {
  REMAINDER = 0
};


template<typename T> ncclDataType_t NCCLType();
template<> ncclDataType_t NCCLType<int32_t>() {
    return ncclInt32;
}
template<> ncclDataType_t NCCLType<int64_t>() {
    return ncclInt64;
}
template<> ncclDataType_t NCCLType<__half>() {
    return ncclHalf;
}
template<> ncclDataType_t NCCLType<float>() {
    return ncclFloat32;
}
template<> ncclDataType_t NCCLType<double>() {
    return ncclFloat64;
}


template<typename IdType> __global__ void _MapProcByRemainder(
    const IdType * const index,
    const int64_t num_index,
    const int64_t num_proc,
    IdType * const proc_id) {
  const int64_t idx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;

  if (idx < num_index) {
    proc_id[idx] = index[idx] % num_proc;
  }
}

template<typename IdType>
__global__ void _MapProcByMaskRemainder(
    const IdType * const index,
    const int64_t num_index,
    const IdType mask,
    IdType * const proc_id) {
  const int64_t idx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;

  if (idx < num_index) {
    proc_id[idx] = index[idx] & mask;
  }
}

template<typename IdType, typename DType>
__global__ void _DualPermKernel(
    const IdType * const in_idx,
    const DType * const in_value,
    const IdType * const perm,
    const int64_t num_in,
    const int64_t num_feat,
    IdType * const out_idx,
    DType * const out_value) {
  // set index permutation
  const int64_t tidx = blockDim.x*static_cast<int64_t>(blockIdx.x)+threadIdx.x;
  if (tidx < num_in) {
    const IdType perm_idx = perm[tidx];
    assert(perm_idx < num_in);
    out_idx[tidx] = in_idx[perm_idx];
  }

  if (num_feat > 1) {
    for (int d = 0; d < blockDim.x; ++d) {
      const int64_t bidx = blockDim.x*static_cast<int64_t>(blockIdx.x) + d;
      if (bidx < num_in) {
        const IdType perm_idx = perm[bidx];
        for (int64_t f = threadIdx.x; f < num_feat; f+=blockDim.x) {
          out_value[bidx*num_feat+f] = in_value[perm_idx*num_feat+f];
        }
      }
    }
  } else {
    if (tidx < num_in) {
      const IdType perm_idx = perm[tidx];
      out_value[tidx] = in_value[perm_idx];
    }
  }
}

template<typename IdType>
__global__ void _ConvertToLocalByRemainder(
    IdType * const items,
    const int64_t num_items,
    const int comm_size) {
  const int64_t idx = threadIdx.x+blockDim.x*blockIdx.x;

  if (idx < num_items) {
    items[idx] = items[idx] / comm_size;
  }
}

template <typename DType, typename IdType>
__global__ void _InversePermKernel(
        const DType* const array,
        const int64_t num_feat,
        int64_t length,
        const IdType* const perm,
        DType* const out) {
  int64_t in_row = blockIdx.x*blockDim.y+threadIdx.y;

  const int64_t stride = blockDim.y*gridDim.x;

  while (in_row < length) {
    int64_t col = threadIdx.x;
    const int64_t out_row = perm[in_row];
    while (col < num_feat) {
      out[out_row*num_feat+col] = array[in_row*num_feat+col];
      col += blockDim.x;
    }
    in_row += stride;
  }
}


}  // namespace

/* NCCLUniqueId **************************************************************/

NCCLUniqueId::NCCLUniqueId() :
  id_() {
  // this ID is unique to the process, not to each call of this function
  NCCL_CALL(ncclGetUniqueId(&id_));
}

ncclUniqueId NCCLUniqueId::Get() const {
  return id_;
}

std::string NCCLUniqueId::ToString() const {
  std::ostringstream oss;

  oss << std::hex;

  for (size_t b = 0; b < NCCL_UNIQUE_ID_BYTES; ++b) {
    const int num = static_cast<uint8_t>(id_.internal[b]);
    oss << std::setw(2) << std::setfill('0') << num;
  }

  std::string result = oss.str();
  CHECK_EQ(result.length(), NCCL_UNIQUE_ID_BYTES*2) <<
    "Invalid NCCL ID format: '" << result << "'";

  return result;
}

void NCCLUniqueId::FromString(
    const std::string& str) {
  // must be exactly 256 hex characters
  CHECK_EQ(str.length(), NCCL_UNIQUE_ID_BYTES * 2) <<
        "Invalid NCCL ID format: '" << str << "'";

  for (size_t b = 0; b < NCCL_UNIQUE_ID_BYTES; ++b) {
    id_.internal[b] = std::strtol(str.substr(b*2, 2).c_str(), nullptr, 16);
  }
}

template<typename IdType>
void GenerateSparseBufferFromRemainder(
    DeviceAPI* const device,
    const DGLContext& ctx,
    const int64_t comm_size,
    const int64_t num_in,
    const IdType * const in_idx,
    IdType * const out_idx,
    IdType * const out_perm,
    int64_t * const out_counts,
    cudaStream_t stream) {
  const int64_t comm_bits =
      static_cast<int64_t>(std::ceil(std::log2(comm_size)));

  // this should only run when we have things to send, otherwise comm_bits
  // will be zero, and several operations will fail
  CHECK_GT(comm_size, 1);

  CUDA_CALL(cudaMemsetAsync(
      out_counts, 0, sizeof(*out_counts)*(comm_size+1), stream));

  if (num_in == 0) {
    // now that we've zero'd out_counts, nothing left to do
    return;
  }

  // First, generate a mapping of indexes to processors
  IdType * proc_id_in = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));
  {
    const dim3 block(256);
    const dim3 grid((num_in+block.x-1)/block.x);

    if (comm_size < (1 << comm_bits)) {
      // comm_size is not a power of 2
      _MapProcByRemainder<<<grid, block, 0, stream>>>(
          in_idx,
          num_in,
          comm_size,
          proc_id_in);
      CUDA_CALL(cudaGetLastError());
    } else {
      // comm_size is a power of 2
      _MapProcByMaskRemainder<<<grid, block, 0, stream>>>(
          in_idx,
          num_in,
          static_cast<IdType>(comm_size-1),  // bit mask
          proc_id_in);
      CUDA_CALL(cudaGetLastError());
    }
  }

  // then create a permutation array that groups processors together by
  // performing a radix sort
  IdType * proc_id_out = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));
  {
    IdArray perm_in = aten::Range(0, num_in, sizeof(IdType)*8, ctx);

    size_t sort_workspace_size;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, sort_workspace_size,
        proc_id_in, proc_id_out, static_cast<IdType*>(perm_in->data), out_perm,
        num_in, 0, comm_bits, stream));

    void * sort_workspace = device->AllocWorkspace(ctx, sort_workspace_size);
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(sort_workspace, sort_workspace_size,
        proc_id_in, proc_id_out, static_cast<IdType*>(perm_in->data), out_perm,
        num_in, 0, comm_bits, stream));
    device->FreeWorkspace(ctx, sort_workspace);
  }
  device->FreeWorkspace(ctx, proc_id_in);

  // finally, permute the input arrays
  // sort the data into continuous buffers for sending
  {
    const dim3 block(256);
    const dim3 grid((num_in+block.x-1)/block.x);

    aten::impl::IndexSelectSingleKernel<<<grid, block, 0, stream>>>(
        in_idx,
        out_perm,
        num_in,
        out_idx);
    CUDA_CALL(cudaGetLastError());
  }

  // Count the number of values to be sent to each processor
  {
    using AtomicCount = unsigned long long; // NOLINT
    static_assert(sizeof(AtomicCount) == sizeof(int64_t),
        "AtomicCount must be the same width as int64_t for atomicAdd "
        "in cub::DeviceHistogram::HistogramEven() to work");

    // TODO(dlasalle): Once https://github.com/NVIDIA/cub/pull/287 is merged,
    // add a compile time check against the cub version to allow
    // num_in > (2 << 31).
    CHECK(num_in < static_cast<int64_t>(std::numeric_limits<int>::max())) <<
        "number of values to insert into histogram must be less than max "
        "value of int.";

    size_t hist_workspace_size;
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        nullptr,
        hist_workspace_size,
        proc_id_out,
        reinterpret_cast<AtomicCount*>(out_counts),
        comm_size+1,
        static_cast<IdType>(0),
        static_cast<IdType>(comm_size+1),
        static_cast<int>(num_in),
        stream));

    void * hist_workspace = device->AllocWorkspace(ctx, hist_workspace_size);
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        hist_workspace,
        hist_workspace_size,
        proc_id_out,
        reinterpret_cast<AtomicCount*>(out_counts),
        comm_size+1,
        static_cast<IdType>(0),
        static_cast<IdType>(comm_size+1),
        static_cast<int>(num_in),
        stream));
    device->FreeWorkspace(ctx, hist_workspace);
  }
  device->FreeWorkspace(ctx, proc_id_out);
}

template<typename IdType, typename DType>
void GenerateSparseBuffersFromRemainder(
    DeviceAPI* const device,
    const DGLContext& ctx,
    const int64_t comm_size,
    const int64_t num_in,
    const int64_t num_feat,
    const IdType * const in_idx,
    const DType * const in_value,
    IdType * const out_idx,
    DType * const out_value,
    int64_t * const out_counts,
    cudaStream_t stream) {
  const int64_t comm_bits =
      static_cast<int64_t>(std::ceil(std::log2(comm_size)));

  // this should only run when we have things to send, otherwise comm_bits
  // will be zero, and several operations will fail
  CHECK_GT(comm_size, 1);

  CUDA_CALL(cudaMemsetAsync(
      out_counts, 0, sizeof(*out_counts)*(comm_size+1), stream));

  if (num_in == 0) {
    // now that we've zero'd out_counts, nothing left to do
    return;
  }

  // First, generate a mapping of indexes to processors
  IdType * proc_id_in = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));
  {
    const dim3 block(256);
    const dim3 grid((num_in+block.x-1)/block.x);

    if (comm_size < (1 << comm_bits)) {
      // comm_size is not a power of 2
      _MapProcByRemainder<<<grid, block, 0, stream>>>(
          in_idx,
          num_in,
          comm_size,
          proc_id_in);
      CUDA_CALL(cudaGetLastError());
    } else {
      // comm_size is a power of 2
      _MapProcByMaskRemainder<<<grid, block, 0, stream>>>(
          in_idx,
          num_in,
          static_cast<IdType>(comm_size-1),  // bit mask
          proc_id_in);
      CUDA_CALL(cudaGetLastError());
    }
  }

  // then create a permutation array that groups processors together by
  // performing a radix sort
  IdType * proc_id_out = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));
  IdType * perm_out = static_cast<IdType*>(device->AllocWorkspace(ctx,
          sizeof(IdType)*num_in));
  {
    IdArray perm_in = aten::Range(0, num_in, sizeof(IdType)*8, ctx);

    size_t sort_workspace_size;
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(nullptr, sort_workspace_size,
        proc_id_in, proc_id_out, static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, comm_bits, stream));

    void * sort_workspace = device->AllocWorkspace(ctx, sort_workspace_size);
    CUDA_CALL(cub::DeviceRadixSort::SortPairs(sort_workspace, sort_workspace_size,
        proc_id_in, proc_id_out, static_cast<IdType*>(perm_in->data), perm_out,
        num_in, 0, comm_bits, stream));
    device->FreeWorkspace(ctx, sort_workspace);
  }
  device->FreeWorkspace(ctx, proc_id_in);

  // perform a histogram and then prefixsum on the sorted proc_id vector

  // finally, permute the input arrays
  // sort the data into continuous buffers for sending
  {
    const dim3 block(256);
    const dim3 grid((num_in+block.x-1)/block.x);

    _DualPermKernel<<<grid, block, 0, stream>>>(
        in_idx,
        in_value,
        perm_out,
        num_in,
        num_feat,
        out_idx,
        out_value);
    CUDA_CALL(cudaGetLastError());
  }
  device->FreeWorkspace(ctx, perm_out);

  // Count the number of values to be sent to each processor
  {
    using AtomicCount = unsigned long long; // NOLINT
    static_assert(sizeof(AtomicCount) == sizeof(int64_t),
        "AtomicCount must be the same width as int64_t for atomicAdd "
        "in cub::DeviceHistogram::HistogramEven() to work");

    // TODO(dlasalle): Once https://github.com/NVIDIA/cub/pull/287 is merged,
    // add a compile time check against the cub version to allow
    // num_in > (2 << 31).
    CHECK(num_in < static_cast<int64_t>(std::numeric_limits<int>::max())) <<
        "number of values to insert into histogram must be less than max "
        "value of int.";

    size_t hist_workspace_size;
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        nullptr,
        hist_workspace_size,
        proc_id_out,
        reinterpret_cast<AtomicCount*>(out_counts),
        comm_size+1,
        static_cast<IdType>(0),
        static_cast<IdType>(comm_size+1),
        static_cast<int>(num_in),
        stream));

    void * hist_workspace = device->AllocWorkspace(ctx, hist_workspace_size);
    CUDA_CALL(cub::DeviceHistogram::HistogramEven(
        hist_workspace,
        hist_workspace_size,
        proc_id_out,
        reinterpret_cast<AtomicCount*>(out_counts),
        comm_size+1,
        static_cast<IdType>(0),
        static_cast<IdType>(comm_size+1),
        static_cast<int>(num_in),
        stream));
    device->FreeWorkspace(ctx, hist_workspace);
  }
  device->FreeWorkspace(ctx, proc_id_out);
}

template<typename IdType, typename DType>
std::pair<IdArray, NDArray> SparsePush(
    NCCLCommunicatorRef comm,
    IdArray in_idx,
    NDArray in_value,
    const int mode_id) {
  CHECK_EQ(in_idx->shape[0], in_value->shape[0]);

  const auto& ctx = in_idx->ctx;
  CHECK_EQ(ctx, in_value->ctx);
  auto device = DeviceAPI::Get(ctx);

  // TODO(dlasalle): Get the stream from the device context.
  cudaStream_t stream = 0;

  CHECK_EQ(in_idx->ndim, 1);

  const int64_t num_in = in_idx->shape[0];
  int64_t num_feat = 1;
  for (int d = 1; d < in_value->ndim; ++d) {
    num_feat *= in_value->shape[d];
  }

  const int64_t comm_size = comm->size();

  if (comm_size == 1) {
    // nothing to do, just return original arrays
    return std::pair<IdArray, NDArray>(in_idx, in_value);
  }

  IdType * send_idx = static_cast<IdType*>(device->AllocWorkspace(ctx,
      num_in*sizeof(IdType)));
  DType * send_value = static_cast<DType*>(device->AllocWorkspace(ctx,
      num_in*num_feat*sizeof(DType)));
  int64_t * send_sum = static_cast<int64_t*>(device->AllocWorkspace(ctx,
      (comm_size+1)*sizeof(int64_t)));

  CHECK_EQ(mode_id, static_cast<int>(AllToAllMode::REMAINDER));
  GenerateSparseBuffersFromRemainder(
      device,
      ctx,
      comm_size,
      num_in,
      num_feat,
      static_cast<const IdType*>(in_idx->data),
      static_cast<const DType*>(in_value->data),
      send_idx,
      send_value,
      send_sum,
      stream);

  // compute the prefix sum of the send values
  int64_t * send_prefix = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_workspace_size,
        send_sum, send_prefix, comm_size+1, stream));

    void * prefix_workspace = device->AllocWorkspace(
        ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_workspace, prefix_workspace_size,
        send_sum, send_prefix, comm_size+1, stream));
    device->FreeWorkspace(ctx, prefix_workspace);
  }

  std::vector<int64_t> send_prefix_host(comm_size+1);
  device->CopyDataFromTo(
      send_prefix,
      0,
      send_prefix_host.data(),
      0,
      send_prefix_host.size()*sizeof(*send_prefix),
      ctx,
      DGLContext{kDLCPU, 0},
      DGLType{kDLInt, sizeof(*send_prefix)*8, 1},
      stream);
  device->FreeWorkspace(ctx, send_prefix);

  CHECK_EQ(send_prefix_host.back(), num_in);

  // communicate the amount to send
  int64_t * recv_sum = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  comm->AllToAll(send_sum, recv_sum, 1, stream);
  device->FreeWorkspace(ctx, send_sum);

  // compute the prefix sum of the recv values
  int64_t * recv_prefix = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_workspace_size,
        recv_sum, recv_prefix, comm_size+1));

    void * prefix_workspace = device->AllocWorkspace(
        ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_workspace, prefix_workspace_size,
        recv_sum, recv_prefix, comm_size+1));
    device->FreeWorkspace(ctx, prefix_workspace);
  }
  device->FreeWorkspace(ctx, recv_sum);

  // finally copy the prefixsum sum down to the host
  std::vector<int64_t> recv_prefix_host(comm_size+1);
  device->CopyDataFromTo(
      recv_prefix,
      0,
      recv_prefix_host.data(),
      0,
      recv_prefix_host.size()*sizeof(*recv_prefix),
      ctx,
      DGLContext{kDLCPU, 0},
      DGLType{kDLInt, sizeof(*recv_prefix)*8, 1},
      stream);
  device->FreeWorkspace(ctx, recv_prefix);

  // use an event to track when copying is done
  cudaEvent_t d2h;
  cudaEventCreate(&d2h);
  cudaEventRecord(d2h, stream);

  // allocate output space
  cudaEventSynchronize(d2h);
  cudaEventDestroy(d2h);

  IdArray recv_idx = aten::NewIdArray(recv_prefix_host.back(), ctx, sizeof(IdType)*8);

  std::vector<int64_t> value_shape(in_value->ndim, 0);
  value_shape[0] = recv_prefix_host.back();
  for (int d = 1; d < in_value->ndim; ++d) {
    value_shape[d] = in_value->shape[d];
  }
  NDArray recv_value = NDArray::Empty(value_shape, in_value->dtype, ctx);

  // send data
  comm->SparseAllToAll(
      send_idx,
      send_value,
      num_feat,
      send_prefix_host.data(),
      static_cast<IdType*>(recv_idx->data),
      static_cast<DType*>(recv_value->data),
      recv_prefix_host.data(),
      stream);
  device->FreeWorkspace(ctx, send_idx);
  device->FreeWorkspace(ctx, send_value);

  return std::pair<IdArray, NDArray>(recv_idx, recv_value);
}

template<typename IdType, typename DType>
NDArray SparsePull(
    NCCLCommunicatorRef comm,
    IdArray req_idx,
    NDArray local_tensor,
    const int mode_id) {
  const auto& ctx = req_idx->ctx;
  CHECK_EQ(ctx, local_tensor->ctx);
  auto device = DeviceAPI::Get(ctx);

  // TODO(dlasalle): Get the stream from the device context.
  cudaStream_t stream = 0;

  CHECK_EQ(req_idx->ndim, 1);

  const int64_t num_in = req_idx->shape[0];
  int64_t num_feat = 1;
  for (int d = 1; d < local_tensor->ndim; ++d) {
    num_feat *= local_tensor->shape[d];
  }

  const int64_t comm_size = comm->size();

  if (comm_size == 1) {
    // Just return index selection from current local_tensor
    return aten::IndexSelect(local_tensor, req_idx);
  }

  // First we need to send our requests to other processors. This means
  // re-ordering our index array to be contiguous among processors, and
  // counting the number of indices we are sending each processor. For now,
  // we assume a poorly partitioned graph, and that there exists the
  // possibility that each processor could request data from this one.

  // the buffer for us to re-order our requests in
  IdType * send_idx = static_cast<IdType*>(device->AllocWorkspace(ctx,
      num_in*sizeof(IdType)));
  IdType * perm = static_cast<IdType*>(
      device->AllocWorkspace(ctx, sizeof(IdType)*num_in));

  // the number of indexes we need to send to each processor
  int64_t * send_sum = static_cast<int64_t*>(device->AllocWorkspace(ctx,
      (comm_size+1)*sizeof(int64_t)));

  CHECK_EQ(mode_id, static_cast<int>(AllToAllMode::REMAINDER));
  GenerateSparseBufferFromRemainder(
      device,
      ctx,
      comm_size,
      num_in,
      static_cast<const IdType*>(req_idx->data),
      send_idx,
      perm,
      send_sum,
      stream);

  // compute the prefix sum of the indexes this process is requesting
  int64_t * request_prefix = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_workspace_size,
        send_sum, request_prefix, comm_size+1, stream));

    void * prefix_workspace = device->AllocWorkspace(
        ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_workspace, prefix_workspace_size,
        send_sum, request_prefix, comm_size+1, stream));
    device->FreeWorkspace(ctx, prefix_workspace);
  }

  std::vector<int64_t> request_prefix_host(comm_size+1);
  device->CopyDataFromTo(
      request_prefix,
      0,
      request_prefix_host.data(),
      0,
      request_prefix_host.size()*sizeof(*request_prefix),
      ctx,
      DGLContext{kDLCPU, 0},
      DGLType{kDLInt, sizeof(*request_prefix)*8, 1},
      stream);
  device->FreeWorkspace(ctx, request_prefix);
  CHECK_EQ(request_prefix_host.back(), num_in);

  // communicate the amount requested
  int64_t * recv_sum = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  comm->AllToAll(send_sum, recv_sum, 1, stream);
  device->FreeWorkspace(ctx, send_sum);

  // compute the prefix sum of the requested indexes
  int64_t * response_prefix = static_cast<int64_t*>(
      device->AllocWorkspace(ctx, sizeof(int64_t)*(comm_size+1)));
  {
    size_t prefix_workspace_size;
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(nullptr, prefix_workspace_size,
        recv_sum, response_prefix, comm_size+1, stream));

    void * prefix_workspace = device->AllocWorkspace(
        ctx, prefix_workspace_size);
    CUDA_CALL(cub::DeviceScan::ExclusiveSum(prefix_workspace, prefix_workspace_size,
        recv_sum, response_prefix, comm_size+1, stream));
    device->FreeWorkspace(ctx, prefix_workspace);
  }
  device->FreeWorkspace(ctx, recv_sum);

  // finally copy the prefixsum sum down to the host
  std::vector<int64_t> response_prefix_host(comm_size+1);
  device->CopyDataFromTo(
      response_prefix,
      0,
      response_prefix_host.data(),
      0,
      response_prefix_host.size()*sizeof(*response_prefix),
      ctx,
      DGLContext{kDLCPU, 0},
      DGLType{kDLInt, sizeof(*response_prefix)*8, 1},
      stream);
  device->FreeWorkspace(ctx, response_prefix);

  // use an event to track when copying is done
  cudaEvent_t d2h;
  cudaEventCreate(&d2h);
  cudaEventRecord(d2h, stream);

  // allocate output space
  cudaEventSynchronize(d2h);
  cudaEventDestroy(d2h);

  // gather requested indexes
  IdType * recv_idx = static_cast<IdType*>(
      device->AllocWorkspace(ctx, response_prefix_host.back()*sizeof(IdType)));
  comm->AllToAllV(
      send_idx,
      request_prefix_host.data(),
      recv_idx,
      response_prefix_host.data(),
      stream);
  device->FreeWorkspace(ctx, send_idx);

  // convert requested indices to local indices depending on partition
  if (response_prefix_host.back() > 0) {
    const dim3 block(128);
    const dim3 grid((response_prefix_host.back()+block.x-1)/block.x);
    _ConvertToLocalByRemainder<<<grid, block, 0, stream>>>(
        recv_idx, response_prefix_host.back(), comm_size);
  }

  // and then index select them into place
  DType * filled_response_value = static_cast<DType*>(device->AllocWorkspace(ctx,
      response_prefix_host.back()*num_feat*sizeof(DType)));
  if (request_prefix_host.back() > 0) {
    dim3 block(256, 1);
    while (block.x >= 2*num_feat) {
        block.x /= 2;
        block.y *= 2;
    }
    const dim3 grid((request_prefix_host.back()+block.y-1)/block.y);

    aten::impl::IndexSelectMultiKernel<<<grid, block, 0, stream>>>(
        static_cast<const DType*>(local_tensor->data),
        num_feat,
        recv_idx,
        response_prefix_host.back(),
        filled_response_value);
    CUDA_CALL(cudaGetLastError());
  }
  device->FreeWorkspace(ctx, recv_idx);

  // we will collect recieved values in this array
  std::vector<int64_t> value_shape(local_tensor->ndim, 0);
  value_shape[0] = request_prefix_host.back();
  for (int d = 1; d < local_tensor->ndim; ++d) {
    value_shape[d] = local_tensor->shape[d];
  }
  DType* filled_request_value = static_cast<DType*>(device->AllocWorkspace(ctx,
      request_prefix_host.back()*num_feat*sizeof(DType)));

  // multiply the prefixes by the number of features being sent
  for (auto& v : request_prefix_host) {
    v *= num_feat;
  }
  for (auto& v : response_prefix_host) {
    v *= num_feat;
  }

  // send the values
  comm->AllToAllV(
      filled_response_value,
      response_prefix_host.data(),
      filled_request_value,
      request_prefix_host.data(),
      stream);
  device->FreeWorkspace(ctx, filled_response_value);

  // finally, we need to permute the values back into the requested order
  NDArray result = NDArray::Empty(value_shape, local_tensor->dtype, ctx);
  if (num_in > 0) {
    dim3 block(256, 1);
    while (block.x >= 2*num_feat) {
        block.x /= 2;
        block.y *= 2;
    }
    const dim3 grid((num_in+block.y-1)/block.y);

    _InversePermKernel<<<grid, block, 0, stream>>>(
        filled_request_value,
        num_feat,
        num_in,
        perm,
        static_cast<DType*>(result->data));
    CUDA_CALL(cudaGetLastError());
  }
  device->FreeWorkspace(ctx, filled_request_value);
  device->FreeWorkspace(ctx, perm);

  return result;
}



/* NCCLCommunicator **********************************************************/

NCCLCommunicator::NCCLCommunicator(
    const int size,
    const int rank,
    ncclUniqueId id) :
  comm_(),
  size_(size),
  rank_(rank) {
  CHECK_LT(rank, size);
  CHECK_GE(rank, 0);

  NCCL_CALL(ncclCommInitRank(&comm_, size_, id, rank_));
}

NCCLCommunicator::~NCCLCommunicator() {
  ncclCommDestroy(comm_);
}

ncclComm_t NCCLCommunicator::Get() {
  return comm_;
}

template<typename DType>
void NCCLCommunicator::AllToAllV(
    const DType * const send,
    const int64_t * const send_prefix,
    DType * const recv,
    const int64_t * const recv_prefix,
    cudaStream_t stream) {
  const ncclDataType_t type = NCCLType<DType>();

  NCCL_CALL(ncclGroupStart());
  for (int r = 0; r < size_; ++r) {
    const int64_t send_size = send_prefix[r+1]-send_prefix[r];
    if (send_size > 0) {
      NCCL_CALL(ncclSend(send+send_prefix[r], send_size, type, r, comm_, stream));
    }
    const int64_t recv_size = recv_prefix[r+1]-recv_prefix[r];
    if (recv_size > 0) {
      NCCL_CALL(ncclRecv(recv+recv_prefix[r], recv_size, type, r, comm_, stream));
    }
  }
  NCCL_CALL(ncclGroupEnd());
}

template
void NCCLCommunicator::AllToAllV<int32_t>(
    const int32_t * const send,
    const int64_t * send_prefix,
    int32_t * const recv,
    const int64_t * recv_prefix,
    cudaStream_t stream);
template
void NCCLCommunicator::AllToAllV<int64_t>(
    const int64_t * const send,
    const int64_t * send_prefix,
    int64_t * const recv,
    const int64_t * recv_prefix,
    cudaStream_t stream);
template
void NCCLCommunicator::AllToAllV<float>(
    const float * const send,
    const int64_t * send_prefix,
    float * const recv,
    const int64_t * recv_prefix,
    cudaStream_t stream);
template
void NCCLCommunicator::AllToAllV<__half>(
    const __half * const send,
    const int64_t * send_prefix,
    __half * const recv,
    const int64_t * recv_prefix,
    cudaStream_t stream);





template<typename IdType>
void NCCLCommunicator::AllToAll(
    const IdType * const send,
    IdType * const recv,
    const int64_t count,
    cudaStream_t stream) {
  const ncclDataType_t type = NCCLType<IdType>();

  ncclGroupStart();
  for (int r = 0; r < size_; ++r) {
    ncclSend(send+(r*count), count, type, r, comm_, stream);
    ncclRecv(recv+(r*count), count, type, r, comm_, stream);
  }
  ncclGroupEnd();
}

template
void NCCLCommunicator::AllToAll<int32_t>(
    const int32_t * const send,
    int32_t * const recv,
    const int64_t count,
    cudaStream_t stream);
template
void NCCLCommunicator::AllToAll<int64_t>(
    const int64_t * const send,
    int64_t * const recv,
    const int64_t count,
    cudaStream_t stream);


template<typename IdType, typename DType>
void NCCLCommunicator::SparseAllToAll(
      const IdType * const send_idx,
      const DType * const send_value,
      const int64_t num_feat,
      const int64_t * const send_prefix,
      IdType * const recv_idx,
      DType * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream) {
  const ncclDataType_t idx_type = NCCLType<IdType>();
  const ncclDataType_t value_type = NCCLType<DType>();

  ncclGroupStart();
  for (int r = 0; r < size_; ++r) {
    const int64_t send_size = send_prefix[r+1]-send_prefix[r];
    if (send_size > 0) {
      ncclSend(send_idx+send_prefix[r], send_size, idx_type, r, comm_, stream);
      ncclSend(send_value+send_prefix[r]*num_feat, send_size*num_feat,
               value_type, r, comm_, stream);
    }
    const int64_t recv_size = recv_prefix[r+1]-recv_prefix[r];
    if (recv_size > 0) {
      ncclRecv(recv_idx+recv_prefix[r], recv_size, idx_type, r, comm_, stream);
      ncclRecv(recv_value+recv_prefix[r]*num_feat, recv_size*num_feat,
               value_type, r, comm_, stream);
    }
  }
  ncclGroupEnd();
}

template
void NCCLCommunicator::SparseAllToAll<int32_t, __half>(
      const int32_t * const send_idx,
      const __half * const send_value,
      const int64_t num_feat,
      const int64_t * const send_prefix,
      int32_t * const recv_idx,
      __half * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream);
template
void NCCLCommunicator::SparseAllToAll<int64_t, __half>(
      const int64_t * const send_idx,
      const __half * const send_value,
      const int64_t num_feat,
      const int64_t * const send_prefix,
      int64_t * const recv_idx,
      __half * const recv_value,
      const int64_t * const recv_prefix,
      cudaStream_t stream);

int NCCLCommunicator::size() const {
  return size_;
}

int NCCLCommunicator::rank() const {
  return rank_;
}


/* CAPI **********************************************************************/

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLGetUniqueId")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  *rv = NCCLUniqueIdRef(std::make_shared<NCCLUniqueId>());
});

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLUniqueIdToString")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NCCLUniqueIdRef idObj = args[0];
  *rv = idObj->ToString();
});

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLUniqueIdFromString")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const std::string str = args[0];

  NCCLUniqueIdRef ref(std::make_shared<NCCLUniqueId>());
  ref->FromString(str);
  *rv = ref;
});

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLCreateComm")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  const int size = args[0];
  const int rank = args[1];
  NCCLUniqueIdRef idObj = args[2];

  *rv = NCCLCommunicatorRef(std::make_shared<NCCLCommunicator>(size, rank,
        idObj->Get()));
});

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLSparseAllToAllPush")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NCCLCommunicatorRef comm = args[0];
  IdArray in_idx = args[1];
  NDArray in_values = args[2];
  const int mode_id = args[3];

  List<ObjectRef> ret;
  ATEN_ID_TYPE_SWITCH(in_idx->dtype, IdType, {
    ATEN_DTYPE_SWITCH(in_values->dtype, DType, "values", {
      auto result = SparsePush<IdType, DType>(comm, in_idx, in_values, mode_id);
      ret.push_back(Value(MakeValue(result.first)));
      ret.push_back(Value(MakeValue(result.second)));
    });
  });

  *rv = ret;
});

DGL_REGISTER_GLOBAL("cuda.nccl._CAPI_DGLNCCLSparseAllToAllPull")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
  NCCLCommunicatorRef comm = args[0];
  // the indexes this process is requesting from others
  IdArray req_idx = args[1];

  // the tensor this process has to fulfill other requests
  NDArray tensor = args[2];
  const int mode_id = args[3];

  ATEN_ID_TYPE_SWITCH(req_idx->dtype, IdType, {
    ATEN_DTYPE_SWITCH(tensor->dtype, DType, "values", {
      *rv = SparsePull<IdType, DType>(comm, req_idx, tensor, mode_id);
    });
  });
});


}  // namespace cuda
}  // namespace runtime
}  // namespace dgl



