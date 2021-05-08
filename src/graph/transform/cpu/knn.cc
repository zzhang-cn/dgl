/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/cpu/knn.cc
 * \brief k-nearest-neighbor (KNN) implementation
 */

#include <vector>
#include <limits>
#include "kdtree_ndarray_adapter.h"
#include "../knn.h"

using namespace dgl::runtime;
using namespace dgl::transform::knn_utils;
namespace dgl {
namespace transform {
namespace impl {

/*! \brief The kd-tree implementation of K-Nearest Neighbors */
template <typename FloatType, typename IdType>
void KdTreeKNN(const NDArray& data_points, const IdArray& data_offsets,
               const NDArray& query_points, const IdArray& query_offsets,
               const int k, IdArray result) {
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];

  for (int64_t b = 0; b < batch_size; ++b) {
    auto d_offset = data_offsets_data[b];
    auto d_length = data_offsets_data[b + 1] - d_offset;
    auto q_offset = query_offsets_data[b];
    auto q_length = query_offsets_data[b + 1] - q_offset;
    auto out_offset = k * q_offset;

    // create view for each segment
    const NDArray current_data_points = const_cast<NDArray*>(&data_points)->CreateView(
      {d_length, feature_size}, data_points->dtype, d_offset * feature_size * sizeof(FloatType));
    const FloatType* current_query_pts_data = query_points_data + q_offset * feature_size;

    KDTreeNDArrayAdapter<FloatType, IdType> kdtree(feature_size, current_data_points);

    // query
    std::vector<IdType> out_buffer(k);
    std::vector<FloatType> out_dist_buffer(k);
#pragma omp parallel for firstprivate(out_buffer) firstprivate(out_dist_buffer)
    for (IdType q = 0; q < q_length; ++q) {
      auto curr_out_offset = k * q + out_offset;
      const FloatType* q_point = current_query_pts_data + q * feature_size;
      size_t num_matches = kdtree.GetIndex()->knnSearch(
        q_point, k, out_buffer.data(), out_dist_buffer.data());

      for (size_t i = 0; i < num_matches; ++i) {
        query_out[curr_out_offset] = q + q_offset;
        data_out[curr_out_offset] = out_buffer[i] + d_offset;
        curr_out_offset++;
      }
    }
  }
}

template <typename FloatType, typename IdType>
void HeapInsert(IdType* out, FloatType* dist,
                IdType new_id, FloatType new_dist,
                int k) {
  // we assume new distance <= worst distance
  IdType left_idx = 0, right_idx = 0, curr_idx = 0, swap_idx = 0;
  while (true) {
    left_idx = 2 * curr_idx + 1;
    right_idx = left_idx + 1;

    if (left_idx >= k) {
      break;
    } else if (right_idx >= k) {
      if (dist[left_idx] > new_dist) {
        swap_idx = left_idx;
      } else {
        break;
      }
    } else {
      if (dist[left_idx] > new_dist && dist[left_idx] > dist[right_idx]) {
        swap_idx = left_idx;
      } else if (dist[right_idx] > new_dist) {
        swap_idx = right_idx;
      } else {
        break;
      }
    }

    dist[curr_idx] = dist[swap_idx];
    out[curr_idx] = out[swap_idx];
    curr_idx = swap_idx;
  }
  dist[curr_idx] = new_dist;
  out[curr_idx] = new_id;
}

template <typename FloatType, typename IdType>
void BruteForceKNN(const NDArray& data_points, const IdArray& data_offsets,
                   const NDArray& query_points, const IdArray& query_offsets,
                   const int k, IdArray result) {
  const int64_t batch_size = data_offsets->shape[0] - 1;
  const int64_t feature_size = data_points->shape[1];
  const IdType* data_offsets_data = data_offsets.Ptr<IdType>();
  const IdType* query_offsets_data = query_offsets.Ptr<IdType>();
  const FloatType* data_points_data = data_points.Ptr<FloatType>();
  const FloatType* query_points_data = query_points.Ptr<FloatType>();
  IdType* query_out = result.Ptr<IdType>();
  IdType* data_out = query_out + k * query_points->shape[0];

  for (int64_t b = 0; b < batch_size; ++b) {
    IdType d_start = data_offsets_data[b], d_end = data_offsets_data[b + 1];
    IdType q_start = query_offsets_data[b], q_end = query_offsets_data[b + 1];

    std::vector<FloatType> dist_buffer(k);

#pragma omp parallel for firstprivate(dist_buffer)
    for (IdType q_idx = q_start; q_idx < q_end; ++q_idx) {
      for (IdType k_idx = 0; k_idx < k; ++k_idx) {
        query_out[q_idx * k + k_idx] = q_idx;
        dist_buffer[k_idx] = std::numeric_limits<FloatType>::max();
      }
      FloatType worst_dist = std::numeric_limits<FloatType>::max();

      for (IdType d_idx = d_start; d_idx < d_end; ++d_idx) {
        FloatType tmp_dist = 0;
        bool early_stop = false;

        // expand loop (x4)
        IdType dim_idx = 0;
        while (dim_idx < feature_size - 3) {
          const FloatType diff0 = query_points_data[q_idx * feature_size + dim_idx]
            - data_points_data[d_idx * feature_size + dim_idx];
          const FloatType diff1 = query_points_data[q_idx * feature_size + dim_idx + 1]
            - data_points_data[d_idx * feature_size + dim_idx + 1];
          const FloatType diff2 = query_points_data[q_idx * feature_size + dim_idx + 2]
            - data_points_data[d_idx * feature_size + dim_idx + 2];
          const FloatType diff3 = query_points_data[q_idx * feature_size + dim_idx + 3]
            - data_points_data[d_idx * feature_size + dim_idx + 3];
          tmp_dist += diff0 * diff0 + diff1 * diff1 + diff2 * diff2 + diff3 * diff3;
          dim_idx += 4;
          if (tmp_dist > worst_dist) {
            early_stop = true;
            dim_idx = feature_size;
            break;
          }
        }

        // last 3 elements
        while (dim_idx < feature_size) {
          const FloatType diff = query_points_data[q_idx * feature_size + dim_idx]
            - data_points_data[d_idx * feature_size + dim_idx];
          tmp_dist += diff * diff;
          ++dim_idx;
          if (tmp_dist > worst_dist) {
            early_stop = true;
            break;
          }
        }

        if (early_stop) continue;

        IdType out_offset = q_idx * k;
        HeapInsert<FloatType, IdType>(
          data_out + out_offset, dist_buffer.data(), d_idx, tmp_dist, k);
        worst_dist = dist_buffer[0];
      }
    }
  }
}

}  // namespace impl

template <DLDeviceType XPU, typename FloatType, typename IdType>
void KNN(const NDArray& data_points, const IdArray& data_offsets,
         const NDArray& query_points, const IdArray& query_offsets,
         const int k, IdArray result, const std::string& algorithm) {
  if (algorithm == std::string("kd-tree")) {
    impl::KdTreeKNN<FloatType, IdType>(
      data_points, data_offsets, query_points, query_offsets, k, result);
  } else if (algorithm == std::string("bruteforce")) {
    impl::BruteForceKNN<FloatType, IdType>(
      data_points, data_offsets, query_points, query_offsets, k, result);
  } else {
    LOG(FATAL) << "Algorithm " << algorithm << " is not supported on CPU";
  }
}

template void KNN<kDLCPU, float, int32_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLCPU, float, int64_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLCPU, double, int32_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
template void KNN<kDLCPU, double, int64_t>(
  const NDArray& data_points, const IdArray& data_offsets,
  const NDArray& query_points, const IdArray& query_offsets,
  const int k, IdArray result, const std::string& algorithm);
}  // namespace transform
}  // namespace dgl
