/*!
 *  Copyright (c) 2021 by Contributors
 * \file graph/sampling/node2vec_impl.h
 * \brief DGL sampler - templated implementation definition of node2vec random
 * walks
 */

#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_IMPL_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_IMPL_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>

#include <functional>
#include <utility>
#include <vector>

namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

/*!
 * \brief Node2vec random walk step function
 */
template <typename IdxType>
using Node2vecStepFunc = std::function<std::pair<dgl_id_t, bool>(
    IdxType *,  // node IDs generated so far
    dgl_id_t,   // last node id generated
    dgl_id_t,   // las last node id generated
    int64_t)>;  // of steps

/*!
 * \brief Node2vec random walk.
 * \param hg The heterograph.
 * \param seeds A 1D array of seed nodes, with the type the source type of the
 * first edge type in the metapath. \param p Float, indicating likelihood of
 * immediately revisiting a node in the walk. \param q Float, control parameter
 * to interpolate between breadth-first strategy and depth-first strategy.
 * \param walk_length Int, length of walk.
 * \param prob A vector of 1D float arrays, indicating the transition
 * probability of each edge by edge type.  An empty float array assumes uniform
 * transition. \return A 2D array of shape (len(seeds), len(walk_length) + 1)
 * with node IDs.  The paths that terminated early are padded with -1.
 */
template <DLDeviceType XPU, typename IdxType>
IdArray Node2vec(const HeteroGraphPtr hg, const IdArray seeds, const double p,
                 const double q, const int64_t walk_length,
                 const FloatArray &prob);

};  // namespace impl

};  // namespace sampling

};  // namespace dgl

#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_IMPL_H_
