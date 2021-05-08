/*!
 *  Copyright (c) 2021 by Contributors
 * \file graph/transform/cpu/kdtree_ndarray_adapter.h
 * \brief NDArray adapter for nanoflann, without
 *        duplicating the storage
 */
#ifndef DGL_GRAPH_TRANSFORM_CPU_KDTREE_NDARRAY_ADAPTER_H_
#define DGL_GRAPH_TRANSFORM_CPU_KDTREE_NDARRAY_ADAPTER_H_

#include <dgl/array.h>
#include <dmlc/logging.h>
#include <nanoflann.hpp>
#include "../../../c_api_common.h"

namespace dgl {
namespace transform {
namespace knn_utils {

/*!
 * \brief A simple 2D NDArray adapter for nanoflann, without duplicating the storage.
 *
 * \tparam FloatType: The type of the point coordinates (typically, double or float).
 * \tparam IdType: The type for indices in the KD-tree index (typically, size_t of int)
 * \tparam FeatureDim: If set to > 0, it specifies a compile-time fixed dimensionality
 *         for the points in the data set, allowing more compiler optimizations.
 * \tparam Dist: The distance metric to use: nanoflann::metric_L1, nanoflann::metric_L2,
 *         nanoflann::metric_L2_Simple, etc.
 * \note The spelling of dgl's adapter ("adapter") is different from naneflann ("adaptor")
 */
template <typename FloatType,
          typename IdType,
          int FeatureDim = -1,
          typename Dist = nanoflann::metric_L2>
class KDTreeNDArrayAdapter {
 public:
  using self_type = KDTreeNDArrayAdapter<FloatType, IdType, FeatureDim, Dist>;
  using metric_type = typename Dist::template traits<FloatType, self_type>::distance_t;
  using index_type = nanoflann::KDTreeSingleIndexAdaptor<
    metric_type, self_type, FeatureDim, IdType>;

  KDTreeNDArrayAdapter(const size_t /* dims */,
                       const NDArray data_points,
                       const int leaf_max_size = 10)
      : data_(data_points) {
    CHECK(data_points->shape[0] != 0 && data_points->shape[1] != 0)
      << "Tensor containing input data point set must be 2D.";
    const size_t dims = data_points->shape[1];
    CHECK(!(FeatureDim > 0 && static_cast<int>(dims) != FeatureDim))
      << "Data set feature dimension does not match the 'FeatureDim' "
      << "template argument.";
    index_ = new index_type(
      static_cast<int>(dims), *this, nanoflann::KDTreeSingleIndexAdaptorParams(leaf_max_size));
    index_->buildIndex();
  }

  ~KDTreeNDArrayAdapter() {
    delete index_;
  }

  index_type* GetIndex() {
    return index_;
  }

  /*!
   * \brief Query for the \a num_closest points to a given point
   *  Note that this is a short-cut method for GetIndex()->findNeighbors().
   */
  void query(const FloatType* query_pt, const size_t num_closest,
             IdType* out_idxs, FloatType* out_dists) const {
    nanoflann::KNNResultSet<FloatType, IdType> resultSet(num_closest);
    resultSet.init(out_idxs, out_dists);
    index_->findNeighbors(resultSet, query_pt, nanoflann::SearchParams());
  }

  /*! \brief Interface expected by KDTreeSingleIndexAdaptor */
  const self_type& derived() const {
    return *this;
  }

  /*! \brief Interface expected by KDTreeSingleIndexAdaptor */
  self_type& derived() {
    return *this;
  }

  /*!
   * \brief Interface expected by KDTreeSingleIndexAdaptor,
   *  return the number of data points
   */
  size_t kdtree_get_point_count() const {
    return data_->shape[0];
  }

  /*!
   * \brief Interface expected by KDTreeSingleIndexAdaptor,
   *  return the dim'th component of the idx'th point
   */
  FloatType kdtree_get_pt(const size_t idx, const size_t dim) const {
    return data_.Ptr<FloatType>()[idx * data_->shape[1] + dim];
  }

  /*!
   * \brief Interface expected by KDTreeSingleIndexAdaptor.
   *  Optional bounding-box computation: return false to
   *  default to a standard bbox computation loop.
   *
   */
  template <typename BBOX>
  bool kdtree_get_bbox(BBOX& /* bb */) const {
    return false;
  }

 private:
  index_type* index_;    // The kd tree index
  const NDArray data_;  // data points
};

}  // namespace knn_utils
}  // namespace transform
}  // namespace dgl

#endif  // DGL_GRAPH_TRANSFORM_CPU_KDTREE_NDARRAY_ADAPTER_H_
