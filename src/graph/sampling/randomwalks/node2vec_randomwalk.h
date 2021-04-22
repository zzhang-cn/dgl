/*!
 *  Copyright (c) 2018 by Contributors
 * \file graph/sampling/node2vec_randomwalk.cc
 * \brief DGL sampler - CPU implementation of node2vec random walk.
 */


#ifndef DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_RANDOMWALK_H_
#define DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_RANDOMWALK_H_

#include <dgl/array.h>
#include <dgl/base_heterograph.h>
#include <dgl/random.h>
#include <utility>
#include <vector>
#include <functional>
#include <cmath>
#include <algorithm>
#include "metapath_randomwalk.h"


namespace dgl {

using namespace dgl::runtime;
using namespace dgl::aten;

namespace sampling {

namespace impl {

namespace {


/*!
 * \brief Random walk step function
 */
template<typename IdxType>
using Node2vecStepFunc = std::function<std::pair<dgl_id_t, bool>(
                IdxType *,    // node IDs generated so far
                dgl_id_t,     // last node id generated
                dgl_id_t,     //
                int64_t)>;    // # of steps

/*
 *
 */
template<typename IdxType>
bool has_edge_between(
        const std::vector<IdArray> &edges,
        dgl_id_t u,
        dgl_id_t v) {
    const IdxType *offsets = static_cast<IdxType *>(edges[0]->data);
    const IdxType *all_succ = static_cast<IdxType *>(edges[1]->data);
    const IdxType *u_succ = all_succ + offsets[u];
    const int64_t size = offsets[u + 1] - offsets[u];

    return std::find(u_succ, u_succ+size, v) != u_succ+size;
}


template<DLDeviceType XPU, typename IdxType>
std::pair<dgl_id_t, bool> Node2vecRandomWalkStep(
        IdxType *data,
        dgl_id_t curr,
        dgl_id_t pre,
        const double p,
        const double q,
        int64_t len,
        const std::vector<IdArray>  &edges,
        const FloatArray &probs,
        TerminatePredicate<IdxType> terminate) {
    const IdxType *offsets = static_cast<IdxType *>(edges[0]->data);
    const IdxType *all_succ = static_cast<IdxType *>(edges[1]->data);
    const IdxType *succ = all_succ + offsets[curr];

    const int64_t size = offsets[curr + 1] - offsets[curr];

    // Isolated node
    if (size == 0)
        return std::make_pair(-1, true);

    IdxType idx = 0;
    // Exist repeated computing problem
    double max_prob = std::max({1/p, 1.0, 1/q});
    double prob0 = 1 / p / max_prob;
    double prob1 = 1 / max_prob;
    double prob2 = 1 / q / max_prob;
    dgl_id_t next_node;
    double r;  // rejection probability.
    if (IsNullArray(probs)) {
        if (len == 0) {
            idx = RandomEngine::ThreadLocal()->RandInt(size);
            next_node = succ[idx];
        } else {
            while (true) {
                idx = RandomEngine::ThreadLocal()->RandInt(size);
                r = RandomEngine::ThreadLocal()->Uniform(0., 1.);
                next_node = succ[idx];
                if (next_node == pre) {
                    if (r < prob0)
                        break;
                } else if (has_edge_between<IdxType>(edges, next_node, pre)) {
                    if (r < prob1)
                        break;
                } else if (r < prob2) {
                    break;
                }
            }
        }
    } else {
        const IdxType *all_eids = static_cast<IdxType *>(edges[2]->data);
        const IdxType *eids = all_eids + offsets[curr];
        FloatArray prob_selected;
        ATEN_FLOAT_TYPE_SWITCH(probs->dtype, DType, "probability", {
            prob_selected = FloatArray::Empty({ size }, probs->dtype, probs->ctx);
            DType *prob_selected_data = static_cast<DType *>(prob_selected->data);
            const DType *prob_etype_data = static_cast<DType *>(probs->data);
            for (int64_t j = 0; j < size; ++j)
                prob_selected_data[j] = prob_etype_data[eids[j]];
        });

        if (len == 0) {
            idx = RandomEngine::ThreadLocal()->Choice<IdxType>(prob_selected);
            next_node = succ[idx];
        } else {
            while (true) {
                idx = RandomEngine::ThreadLocal()->Choice<IdxType>(prob_selected);
                r = RandomEngine::ThreadLocal()->Uniform(0., 1.);
                next_node = succ[idx];
                if (next_node == pre) {
                    if (r < prob0)
                        break;
                } else if (has_edge_between<IdxType>(edges, next_node, pre)) {
                    if (r < prob1)
                        break;
                } else if (r < prob2) {
                    break;
                }
            }
        }
    }
    curr = next_node;

    return std::make_pair(curr, terminate(data, curr, len));
}

template<DLDeviceType XPU, typename IdxType>
IdArray Node2vecGenericRandomWalk(
        const IdArray seeds,
        int64_t walk_length,
        Node2vecStepFunc<IdxType> step) {
    int64_t num_seeds = seeds->shape[0];
    walk_length = walk_length + 1;  //

    IdArray traces = IdArray::Empty({num_seeds, walk_length}, seeds->dtype, seeds->ctx);

    const IdxType *seed_data = static_cast<IdxType *>(seeds->data);
    IdxType *traces_data = static_cast<IdxType *>(traces->data);

#pragma omp parallel for
    for (int64_t seed_id = 0; seed_id < num_seeds; ++seed_id) {
        int64_t i;
        dgl_id_t curr = seed_data[seed_id];
        dgl_id_t pre = curr;
        traces_data[seed_id * walk_length] = curr;

        for (i = 0; i < walk_length; ++i) {
            const auto &succ = step(traces_data + seed_id * walk_length, curr, pre, i);
            pre = curr;
            curr = succ.first;
            traces_data[seed_id*walk_length+i+1] = curr;
            if (succ.second)
                break;
        }

        for (; i < walk_length; ++i)
            traces_data[seed_id * walk_length + i + 1] = -1;
    }
    return traces;
}

template<DLDeviceType XPU, typename IdxType>
IdArray Node2vecRandomWalk(
        const HeteroGraphPtr g,
        const IdArray seeds,
        const double p,
        const double q,
        const int64_t walk_length,
        const FloatArray &prob,
        TerminatePredicate<IdxType> terminate){
    std::vector<IdArray> edges;
    edges = g->GetAdj(0, true, "csr");  // homogeneous graph.

    Node2vecStepFunc<IdxType > step =
        [&edges, &prob, p, q, terminate]
        (IdxType *data, dgl_id_t curr, dgl_id_t pre, int64_t len){
            return Node2vecRandomWalkStep<XPU, IdxType>(
                data, curr, pre, p, q, len, edges, prob, terminate);
    };

    return Node2vecGenericRandomWalk<XPU, IdxType>(seeds, walk_length, step);
}

};  // namespace

};  // namespace impl

};  // namespace sampling

};  // namespace dgl
#endif  // DGL_GRAPH_SAMPLING_RANDOMWALKS_NODE2VEC_RANDOMWALK_H_
