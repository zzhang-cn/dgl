/*!
 *  Copyright (c) 2019 by Contributors
 * \file graph/transform/knn.cc
 * \brief k-nearest-neighbor (KNN) implementation
 */

#include <dgl/runtime/registry.h>
#include <dgl/runtime/packed_func.h>
#include <vector>
#include "kdtree_ndarray_adapter.h"
#include "knn.h"
#include "../../array/check.h"

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
  int64_t batch_size = data_offsets->shape[0] - 1;
  int64_t feature_size = data_points->shape[1];
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
    for (int64_t q = 0; q < q_length; ++q) {
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
}  // namespace impl

template <DLDeviceType XPU, typename FloatType, typename IdType>
void KNN(const NDArray& data_points, const IdArray& data_offsets,
         const NDArray& query_points, const IdArray& query_offsets,
         const int k, IdArray result, const std::string& algorithm) {
  if (algorithm == std::string("kd-tree")) {
    impl::KdTreeKNN<FloatType, IdType>(
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

DGL_REGISTER_GLOBAL("transform._CAPI_DGLKNN")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const NDArray data_points = args[0];
    const IdArray data_offsets = args[1];
    const NDArray query_points = args[2];
    const IdArray query_offsets = args[3];
    const int k = args[4];
    IdArray result = args[5];
    const std::string algorithm = args[6];

    aten::CheckContiguous(
      {data_points, data_offsets, query_points, query_offsets, result},
      {"data_points", "data_offsets", "query_points", "query_offsets", "result"});
    aten::CheckCtx(
      data_points->ctx, {data_offsets, query_points, query_offsets, result},
      {"data_offsets", "query_points", "query_offsets", "result"});

    ATEN_XPU_SWITCH(data_points->ctx.device_type, XPU, "KNN", {
      ATEN_FLOAT_TYPE_SWITCH(data_points->dtype, FloatType, "data_points", {
        ATEN_ID_TYPE_SWITCH(result->dtype, IdType, {
          KNN<XPU, FloatType, IdType>(
            data_points, data_offsets, query_points,
            query_offsets, k, result, algorithm);
        });
      });
    });
  });

}  // namespace transform
}  // namespace dgl
