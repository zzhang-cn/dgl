/*!
 *  Copyright (c) 2021 by Contributors
 * \file graph/transform/knn.h
 * \brief k-nearest-neighbor (KNN) implementation
 */

#ifndef DGL_GRAPH_TRANSFORM_KNN_H_
#define DGL_GRAPH_TRANSFORM_KNN_H_

#include <dgl/array.h>
#include <string>

namespace dgl {
namespace transform {

/*!
 * \brief For each point in each segment in \a query_points, find \a k nearest
 *  points in the same segment in \a data_points. \a data_offsets and \a query_offsets
 *  determine the start index of each segment in \a data_points and \a query_points.
 *
 * \param data_points dataset points
 * \param data_offsets offsets of point index in \a data_points
 * \param query_points query points
 * \param query_offsets offsets of point index in \a query_points
 * \param k the number of nearest points
 * \param result output array
 * \param algorithm algorithm used to compute the k-nearest neighbors
 *
 * \return A 2D tensor indicating the index relation between \a query_points and \a data_points.
 */
template <DLDeviceType XPU, typename FloatType, typename IdType>
void KNN(const NDArray& data_points, const IdArray& data_offsets,
         const NDArray& query_points, const IdArray& query_offsets,
         const int k, IdArray result, const std::string& algorithm);

}  // namespace transform
}  // namespace dgl


#endif  // DGL_GRAPH_TRANSFORM_KNN_H_
