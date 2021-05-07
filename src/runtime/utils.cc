/*!
 *  Copyright (c) 2020 by Contributors
 * \file utils.cc
 * \brief DGL util functions
 */

#include <dmlc/omp.h>


#include <dgl/aten/coo.h>
#include <dgl/packed_func_ext.h>
#include <utility>

#include "../c_api_common.h"
#include "../array/array_op.h"


using namespace dgl::runtime;
using namespace dgl::aten::impl;

namespace dgl {

DGL_REGISTER_GLOBAL("utils.internal._CAPI_DGLSetOMPThreads")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    int num_threads = args[0];
    omp_set_num_threads(num_threads);
  });


DGL_REGISTER_GLOBAL("utils.checks._CAPI_DGLCOOIsSorted")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    IdArray src = args[0];
    IdArray dst = args[1];
    int64_t num_src = args[2];
    int64_t num_dst = args[3];

    bool row_sorted, col_sorted;
    std::tie(row_sorted, col_sorted) = COOIsSorted(
        aten::COOMatrix(num_src, num_dst, src, dst));

    // make sure col_sorted is only true when row_sorted is true
    assert(!(!row_sorted && col_sorted));

    // 0 for unosrted, 1 for row sorted, 2 for row and col sorted
    int64_t sorted_status = row_sorted + col_sorted;
    *rv = sorted_status;
  });

}  // namespace dgl
