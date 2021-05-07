/*!
 *  Copyright (c) 2021 by Contributors
 * \file array/cpu/array_index_select.cuh
 * \brief Array index select GPU kernel implementation
 */

#ifndef DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_CUH_
#define DGL_ARRAY_CUDA_ARRAY_INDEX_SELECT_CUH_

namespace dgl {
namespace aten {
namespace impl {

template <typename DType, typename IdType>
__global__ void IndexSelectSingleKernel(const DType* array, const IdType* index,
                                   int64_t length, DType* out) {
  int tx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride_x = gridDim.x * blockDim.x;
  while (tx < length) {
    out[tx] = array[index[tx]];
    tx += stride_x;
  }
}

template <typename DType, typename IdType>
__global__ void IndexSelectMultiKernel(
        const DType* const array, 
        const int64_t num_feat,
        const IdType* const index,
        const int64_t length,
        DType* const out) {
  int64_t out_row = blockIdx.x*blockDim.y+threadIdx.y;

  const int64_t stride = blockDim.y*gridDim.x;

  while (out_row < length) {
    int64_t col = threadIdx.x;
    const int64_t in_row = index[out_row];
    while (col < num_feat) {
      out[out_row*num_feat+col] = array[in_row*num_feat+col];
      col += blockDim.x;
    }
    out_row += stride;
  }
}

}  // namespace impl
}  // namespace aten
}  // namespace dgl

#endif
