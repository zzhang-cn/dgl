/*!
 *  Copyright (c) 2019 by Contributors
 * \file array/array.cc
 * \brief DGL array utilities implementation
 */
#include <dgl/array.h>
#include <dgl/graph_traversal.h>
#include <dgl/packed_func_ext.h>
#include <dgl/runtime/container.h>
#include <dgl/runtime/shared_mem.h>
#include <dgl/runtime/device_api.h>
#include <sstream>
#include "../c_api_common.h"
#include "./array_op.h"
#include "./arith.h"

using namespace dgl::runtime;

namespace dgl {
namespace aten {

IdArray NewIdArray(int64_t length, DLContext ctx, uint8_t nbits) {
  return IdArray::Empty({length}, DLDataType{kDLInt, nbits, 1}, ctx);
}

IdArray Clone(IdArray arr) {
  IdArray ret = NewIdArray(arr->shape[0], arr->ctx, arr->dtype.bits);
  ret.CopyFrom(arr);
  return ret;
}

IdArray Range(int64_t low, int64_t high, uint8_t nbits, DLContext ctx) {
  IdArray ret;
  ATEN_XPU_SWITCH_CUDA(ctx.device_type, XPU, "Range", {
    if (nbits == 32) {
      ret = impl::Range<XPU, int32_t>(low, high, ctx);
    } else if (nbits == 64) {
      ret = impl::Range<XPU, int64_t>(low, high, ctx);
    } else {
      LOG(FATAL) << "Only int32 or int64 is supported.";
    }
  });
  return ret;
}

IdArray Full(int64_t val, int64_t length, uint8_t nbits, DLContext ctx) {
  IdArray ret;
  ATEN_XPU_SWITCH_CUDA(ctx.device_type, XPU, "Full", {
    if (nbits == 32) {
      ret = impl::Full<XPU, int32_t>(val, length, ctx);
    } else if (nbits == 64) {
      ret = impl::Full<XPU, int64_t>(val, length, ctx);
    } else {
      LOG(FATAL) << "Only int32 or int64 is supported.";
    }
  });
  return ret;
}

IdArray AsNumBits(IdArray arr, uint8_t bits) {
  CHECK(bits == 32 || bits == 64)
    << "Invalid ID type. Must be int32 or int64, but got int"
    << static_cast<int>(bits) << ".";
  if (arr->dtype.bits == bits)
    return arr;
  if (arr.NumElements() == 0)
    return NewIdArray(arr->shape[0], arr->ctx, bits);
  IdArray ret;
  ATEN_XPU_SWITCH_CUDA(arr->ctx.device_type, XPU, "AsNumBits", {
    ATEN_ID_TYPE_SWITCH(arr->dtype, IdType, {
      ret = impl::AsNumBits<XPU, IdType>(arr, bits);
    });
  });
  return ret;
}

IdArray HStack(IdArray lhs, IdArray rhs) {
  IdArray ret;
  CHECK_SAME_CONTEXT(lhs, rhs);
  CHECK_SAME_DTYPE(lhs, rhs);
  CHECK_EQ(lhs->shape[0], rhs->shape[0]);
  auto device = runtime::DeviceAPI::Get(lhs->ctx);
  const auto& ctx = lhs->ctx;
  ATEN_ID_TYPE_SWITCH(lhs->dtype, IdType, {
    const int64_t len = lhs->shape[0];
    ret = NewIdArray(2 * len, lhs->ctx, lhs->dtype.bits);
    device->CopyDataFromTo(lhs.Ptr<IdType>(), 0,
                           ret.Ptr<IdType>(), 0,
                           len * sizeof(IdType),
                           ctx, ctx, lhs->dtype, nullptr);
    device->CopyDataFromTo(rhs.Ptr<IdType>(), 0,
                           ret.Ptr<IdType>(), len * sizeof(IdType),
                           len * sizeof(IdType),
                           ctx, ctx, lhs->dtype, nullptr);
  });
  return ret;
}

NDArray IndexSelect(NDArray array, IdArray index) {
  NDArray ret;
  CHECK_SAME_CONTEXT(array, index);
  CHECK_GE(array->ndim, 1) << "Only support array with at least 1 dimension";
  CHECK_EQ(array->shape[0], array.NumElements()) << "Only support tensor"
    << " whose first dimension equals number of elements, e.g. (5,), (5, 1)";
  CHECK_EQ(index->ndim, 1) << "Index array must be an 1D array.";
  ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "IndexSelect", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
        ret = impl::IndexSelect<XPU, DType, IdType>(array, index);
      });
    });
  });
  return ret;
}

template<typename ValueType>
ValueType IndexSelect(NDArray array, int64_t index) {
  CHECK_EQ(array->ndim, 1) << "Only support select values from 1D array.";
  CHECK(index >= 0 && index < array.NumElements())
    << "Index " << index << " is out of bound.";
  ValueType ret = 0;
  ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "IndexSelect", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ret = impl::IndexSelect<XPU, DType>(array, index);
    });
  });
  return ret;
}
template int32_t IndexSelect<int32_t>(NDArray array, int64_t index);
template int64_t IndexSelect<int64_t>(NDArray array, int64_t index);
template uint32_t IndexSelect<uint32_t>(NDArray array, int64_t index);
template uint64_t IndexSelect<uint64_t>(NDArray array, int64_t index);
template float IndexSelect<float>(NDArray array, int64_t index);
template double IndexSelect<double>(NDArray array, int64_t index);

NDArray IndexSelect(NDArray array, int64_t start, int64_t end) {
  CHECK_EQ(array->ndim, 1) << "Only support select values from 1D array.";
  CHECK(start >= 0 && start < array.NumElements())
    << "Index " << start << " is out of bound.";
  CHECK(end >= 0 && end <= array.NumElements())
    << "Index " << end << " is out of bound.";
  CHECK_LE(start, end);
  auto device = runtime::DeviceAPI::Get(array->ctx);
  const int64_t len = end - start;
  NDArray ret = NDArray::Empty({len}, array->dtype, array->ctx);
  ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
    device->CopyDataFromTo(array->data, start * sizeof(DType),
                           ret->data, 0, len * sizeof(DType),
                           array->ctx, ret->ctx, array->dtype, nullptr);
  });
  return ret;
}

NDArray Scatter(NDArray array, IdArray indices) {
  NDArray ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, "Scatter", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(indices->dtype, IdType, {
        ret = impl::Scatter<XPU, DType, IdType>(array, indices);
      });
    });
  });
  return ret;
}

void Scatter_(IdArray index, NDArray value, NDArray out) {
  CHECK_SAME_DTYPE(value, out);
  CHECK_SAME_CONTEXT(index, value);
  CHECK_SAME_CONTEXT(index, out);
  CHECK_EQ(value->shape[0], index->shape[0]);
  if (index->shape[0] == 0)
    return;
  ATEN_XPU_SWITCH_CUDA(value->ctx.device_type, XPU, "Scatter_", {
    ATEN_DTYPE_SWITCH(value->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(index->dtype, IdType, {
        impl::Scatter_<XPU, DType, IdType>(index, value, out);
      });
    });
  });
}

NDArray Repeat(NDArray array, IdArray repeats) {
  NDArray ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, "Repeat", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "values", {
      ATEN_ID_TYPE_SWITCH(repeats->dtype, IdType, {
        ret = impl::Repeat<XPU, DType, IdType>(array, repeats);
      });
    });
  });
  return ret;
}

IdArray Relabel_(const std::vector<IdArray>& arrays) {
  IdArray ret;
  ATEN_XPU_SWITCH(arrays[0]->ctx.device_type, XPU, "Relabel_", {
    ATEN_ID_TYPE_SWITCH(arrays[0]->dtype, IdType, {
      ret = impl::Relabel_<XPU, IdType>(arrays);
    });
  });
  return ret;
}

NDArray Concat(const std::vector<IdArray>& arrays) {
  IdArray ret;

  int64_t len = 0, offset = 0;
  for (size_t i = 0; i < arrays.size(); ++i) {
    len += arrays[i]->shape[0];
    CHECK_SAME_DTYPE(arrays[0], arrays[i]);
    CHECK_SAME_CONTEXT(arrays[0], arrays[i]);
  }

  NDArray ret_arr = NDArray::Empty({len},
                                   arrays[0]->dtype,
                                   arrays[0]->ctx);

  auto device = runtime::DeviceAPI::Get(arrays[0]->ctx);
  for (size_t i = 0; i < arrays.size(); ++i) {
    ATEN_DTYPE_SWITCH(arrays[i]->dtype, DType, "array", {
      device->CopyDataFromTo(
        static_cast<DType*>(arrays[i]->data),
        0,
        static_cast<DType*>(ret_arr->data),
        offset,
        arrays[i]->shape[0] * sizeof(DType),
        arrays[i]->ctx,
        ret_arr->ctx,
        arrays[i]->dtype,
        nullptr);

        offset += arrays[i]->shape[0] * sizeof(DType);
    });
  }

  return ret_arr;
}

template<typename ValueType>
std::tuple<NDArray, IdArray, IdArray> Pack(NDArray array, ValueType pad_value) {
  std::tuple<NDArray, IdArray, IdArray> ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, "Pack", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "array", {
      ret = impl::Pack<XPU, DType>(array, static_cast<DType>(pad_value));
    });
  });
  return ret;
}

template std::tuple<NDArray, IdArray, IdArray> Pack<int32_t>(NDArray, int32_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<int64_t>(NDArray, int64_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<uint32_t>(NDArray, uint32_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<uint64_t>(NDArray, uint64_t);
template std::tuple<NDArray, IdArray, IdArray> Pack<float>(NDArray, float);
template std::tuple<NDArray, IdArray, IdArray> Pack<double>(NDArray, double);

std::pair<NDArray, IdArray> ConcatSlices(NDArray array, IdArray lengths) {
  std::pair<NDArray, IdArray> ret;
  ATEN_XPU_SWITCH(array->ctx.device_type, XPU, "ConcatSlices", {
    ATEN_DTYPE_SWITCH(array->dtype, DType, "array", {
      ATEN_ID_TYPE_SWITCH(lengths->dtype, IdType, {
        ret = impl::ConcatSlices<XPU, DType, IdType>(array, lengths);
      });
    });
  });
  return ret;
}

IdArray CumSum(IdArray array, bool prepend_zero) {
  IdArray ret;
  ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "CumSum", {
    ATEN_ID_TYPE_SWITCH(array->dtype, IdType, {
      ret = impl::CumSum<XPU, IdType>(array, prepend_zero);
    });
  });
  return ret;
}

IdArray NonZero(NDArray array) {
  IdArray ret;
  ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "NonZero", {
    ATEN_ID_TYPE_SWITCH(array->dtype, DType, {
      ret = impl::NonZero<XPU, DType>(array);
    });
  });
  return ret;
}

std::pair<IdArray, IdArray> Sort(IdArray array, const int num_bits) {
  if (array.NumElements() == 0) {
    IdArray idx = NewIdArray(0, array->ctx, 64);
    return std::make_pair(array, idx);
  }
  std::pair<IdArray, IdArray> ret;
  ATEN_XPU_SWITCH_CUDA(array->ctx.device_type, XPU, "Sort", {
    ATEN_ID_TYPE_SWITCH(array->dtype, IdType, {
      ret = impl::Sort<XPU, IdType>(array, num_bits);
    });
  });
  return ret;
}

std::string ToDebugString(NDArray array) {
  std::ostringstream oss;
  NDArray a = array.CopyTo(DLContext{kDLCPU, 0});
  oss << "array([";
  ATEN_DTYPE_SWITCH(a->dtype, DType, "array", {
    for (int64_t i = 0; i < std::min<int64_t>(a.NumElements(), 10L); ++i) {
      oss << a.Ptr<DType>()[i] << ", ";
    }
  });
  if (a.NumElements() > 10)
    oss << "...";
  oss << "], dtype=" << array->dtype << ", ctx=" << array->ctx << ")";
  return oss.str();
}

///////////////////////// CSR routines //////////////////////////

bool CSRIsNonZero(CSRMatrix csr, int64_t row, int64_t col) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  CHECK(col >= 0 && col < csr.num_cols) << "Invalid col index: " << col;
  bool ret = false;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRIsNonZero", {
    ret = impl::CSRIsNonZero<XPU, IdType>(csr, row, col);
  });
  return ret;
}

NDArray CSRIsNonZero(CSRMatrix csr, NDArray row, NDArray col) {
  NDArray ret;
  CHECK_SAME_DTYPE(csr.indices, row);
  CHECK_SAME_DTYPE(csr.indices, col);
  CHECK_SAME_CONTEXT(csr.indices, row);
  CHECK_SAME_CONTEXT(csr.indices, col);
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRIsNonZero", {
    ret = impl::CSRIsNonZero<XPU, IdType>(csr, row, col);
  });
  return ret;
}

bool CSRHasDuplicate(CSRMatrix csr) {
  bool ret = false;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRHasDuplicate", {
    ret = impl::CSRHasDuplicate<XPU, IdType>(csr);
  });
  return ret;
}

int64_t CSRGetRowNNZ(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  int64_t ret = 0;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetRowNNZ", {
    ret = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  });
  return ret;
}

NDArray CSRGetRowNNZ(CSRMatrix csr, NDArray row) {
  NDArray ret;
  CHECK_SAME_DTYPE(csr.indices, row);
  CHECK_SAME_CONTEXT(csr.indices, row);
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetRowNNZ", {
    ret = impl::CSRGetRowNNZ<XPU, IdType>(csr, row);
  });
  return ret;
}

NDArray CSRGetRowColumnIndices(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  NDArray ret;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetRowColumnIndices", {
    ret = impl::CSRGetRowColumnIndices<XPU, IdType>(csr, row);
  });
  return ret;
}

NDArray CSRGetRowData(CSRMatrix csr, int64_t row) {
  CHECK(row >= 0 && row < csr.num_rows) << "Invalid row index: " << row;
  NDArray ret;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetRowData", {
    ret = impl::CSRGetRowData<XPU, IdType>(csr, row);
  });
  return ret;
}

bool CSRIsSorted(CSRMatrix csr) {
  if (csr.indices->shape[0] <= 1)
    return true;
  bool ret = false;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRIsSorted", {
    ret = impl::CSRIsSorted<XPU, IdType>(csr);
  });
  return ret;
}

NDArray CSRGetData(CSRMatrix csr, NDArray rows, NDArray cols) {
  NDArray ret;
  CHECK_SAME_DTYPE(csr.indices, rows);
  CHECK_SAME_DTYPE(csr.indices, cols);
  CHECK_SAME_CONTEXT(csr.indices, rows);
  CHECK_SAME_CONTEXT(csr.indices, cols);
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetData", {
    ret = impl::CSRGetData<XPU, IdType>(csr, rows, cols);
  });
  return ret;
}

std::vector<NDArray> CSRGetDataAndIndices(
    CSRMatrix csr, NDArray rows, NDArray cols) {
  CHECK_SAME_DTYPE(csr.indices, rows);
  CHECK_SAME_DTYPE(csr.indices, cols);
  CHECK_SAME_CONTEXT(csr.indices, rows);
  CHECK_SAME_CONTEXT(csr.indices, cols);
  std::vector<NDArray> ret;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRGetDataAndIndices", {
    ret = impl::CSRGetDataAndIndices<XPU, IdType>(csr, rows, cols);
  });
  return ret;
}

CSRMatrix CSRTranspose(CSRMatrix csr) {
  CSRMatrix ret;
  ATEN_XPU_SWITCH_CUDA(csr.indptr->ctx.device_type, XPU, "CSRTranspose", {
    ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
      ret = impl::CSRTranspose<XPU, IdType>(csr);
    });
  });
  return ret;
}

COOMatrix CSRToCOO(CSRMatrix csr, bool data_as_order) {
  COOMatrix ret;
  if (data_as_order) {
    ATEN_XPU_SWITCH_CUDA(csr.indptr->ctx.device_type, XPU, "CSRToCOODataAsOrder", {
      ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
        ret = impl::CSRToCOODataAsOrder<XPU, IdType>(csr);
      });
    });
  } else {
    ATEN_XPU_SWITCH_CUDA(csr.indptr->ctx.device_type, XPU, "CSRToCOO", {
      ATEN_ID_TYPE_SWITCH(csr.indptr->dtype, IdType, {
        ret = impl::CSRToCOO<XPU, IdType>(csr);
      });
    });
  }
  return ret;
}

CSRMatrix CSRSliceRows(CSRMatrix csr, int64_t start, int64_t end) {
  CHECK(start >= 0 && start < csr.num_rows) << "Invalid start index: " << start;
  CHECK(end >= 0 && end <= csr.num_rows) << "Invalid end index: " << end;
  CHECK_GE(end, start);
  CSRMatrix ret;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRSliceRows", {
    ret = impl::CSRSliceRows<XPU, IdType>(csr, start, end);
  });
  return ret;
}

CSRMatrix CSRSliceRows(CSRMatrix csr, NDArray rows) {
  CHECK_SAME_DTYPE(csr.indices, rows);
  CHECK_SAME_CONTEXT(csr.indices, rows);
  CSRMatrix ret;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRSliceRows", {
    ret = impl::CSRSliceRows<XPU, IdType>(csr, rows);
  });
  return ret;
}

CSRMatrix CSRSliceMatrix(CSRMatrix csr, NDArray rows, NDArray cols) {
  CHECK_SAME_DTYPE(csr.indices, rows);
  CHECK_SAME_DTYPE(csr.indices, cols);
  CHECK_SAME_CONTEXT(csr.indices, rows);
  CHECK_SAME_CONTEXT(csr.indices, cols);
  CSRMatrix ret;
  ATEN_CSR_SWITCH_CUDA(csr, XPU, IdType, "CSRSliceMatrix", {
    ret = impl::CSRSliceMatrix<XPU, IdType>(csr, rows, cols);
  });
  return ret;
}

void CSRSort_(CSRMatrix* csr) {
  if (csr->sorted)
    return;
  ATEN_CSR_SWITCH_CUDA(*csr, XPU, IdType, "CSRSort_", {
    impl::CSRSort_<XPU, IdType>(csr);
  });
}

CSRMatrix CSRReorder(CSRMatrix csr, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids) {
  CSRMatrix ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, "CSRReorder", {
    ret = impl::CSRReorder<XPU, IdType>(csr, new_row_ids, new_col_ids);
  });
  return ret;
}

CSRMatrix CSRRemove(CSRMatrix csr, IdArray entries) {
  CSRMatrix ret;
  ATEN_CSR_SWITCH(csr, XPU, IdType, "CSRRemove", {
    ret = impl::CSRRemove<XPU, IdType>(csr, entries);
  });
  return ret;
}

COOMatrix CSRRowWiseSampling(
    CSRMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob, bool replace) {
  COOMatrix ret;
  if (IsNullArray(prob)) {
    ATEN_CSR_SWITCH_CUDA(mat, XPU, IdType, "CSRRowWiseSampling", {
      ret = impl::CSRRowWiseSamplingUniform<XPU, IdType>(mat, rows, num_samples, replace);
    });
  } else {
    ATEN_CSR_SWITCH(mat, XPU, IdType, "CSRRowWiseSampling", {
      ATEN_FLOAT_TYPE_SWITCH(prob->dtype, FloatType, "probability", {
        ret = impl::CSRRowWiseSampling<XPU, IdType, FloatType>(
            mat, rows, num_samples, prob, replace);
      });
    });
  }
  return ret;
}

COOMatrix CSRRowWiseTopk(
    CSRMatrix mat, IdArray rows, int64_t k, NDArray weight, bool ascending) {
  COOMatrix ret;
  ATEN_CSR_SWITCH(mat, XPU, IdType, "CSRRowWiseTopk", {
    ATEN_DTYPE_SWITCH(weight->dtype, DType, "weight", {
      ret = impl::CSRRowWiseTopk<XPU, IdType, DType>(
          mat, rows, k, weight, ascending);
    });
  });
  return ret;
}


CSRMatrix UnionCsr(const std::vector<CSRMatrix>& csrs) {
  CSRMatrix ret;
  CHECK_GT(csrs.size(), 1) << "UnionCsr creates a union of multiple CSRMatrixes";
  // sanity check
  for (size_t i = 1; i < csrs.size(); ++i) {
    CHECK_EQ(csrs[0].num_rows, csrs[i].num_rows) <<
      "UnionCsr requires both CSRMatrix have same number of rows";
    CHECK_EQ(csrs[0].num_cols, csrs[i].num_cols) <<
      "UnionCsr requires both CSRMatrix have same number of cols";
    CHECK_SAME_CONTEXT(csrs[0].indptr, csrs[i].indptr);
    CHECK_SAME_DTYPE(csrs[0].indptr, csrs[i].indptr);
  }

  ATEN_CSR_SWITCH(csrs[0], XPU, IdType, "UnionCsr", {
    ret = impl::UnionCsr<XPU, IdType>(csrs);
  });
  return ret;
}


std::tuple<CSRMatrix, IdArray, IdArray>
CSRToSimple(const CSRMatrix& csr) {
  std::tuple<CSRMatrix, IdArray, IdArray> ret;

  CSRMatrix sorted_csr = (CSRIsSorted(csr)) ? csr : CSRSort(csr);
  ATEN_CSR_SWITCH(csr, XPU, IdType, "CSRToSimple", {
    ret = impl::CSRToSimple<XPU, IdType>(sorted_csr);
  });
  return ret;
}

///////////////////////// COO routines //////////////////////////

bool COOIsNonZero(COOMatrix coo, int64_t row, int64_t col) {
  bool ret = false;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOIsNonZero", {
    ret = impl::COOIsNonZero<XPU, IdType>(coo, row, col);
  });
  return ret;
}

NDArray COOIsNonZero(COOMatrix coo, NDArray row, NDArray col) {
  NDArray ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOIsNonZero", {
    ret = impl::COOIsNonZero<XPU, IdType>(coo, row, col);
  });
  return ret;
}

bool COOHasDuplicate(COOMatrix coo) {
  bool ret = false;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOHasDuplicate", {
    ret = impl::COOHasDuplicate<XPU, IdType>(coo);
  });
  return ret;
}

int64_t COOGetRowNNZ(COOMatrix coo, int64_t row) {
  int64_t ret = 0;
  ATEN_COO_SWITCH_CUDA(coo, XPU, IdType, "COOGetRowNNZ", {
    ret = impl::COOGetRowNNZ<XPU, IdType>(coo, row);
  });
  return ret;
}

NDArray COOGetRowNNZ(COOMatrix coo, NDArray row) {
  NDArray ret;
  ATEN_COO_SWITCH_CUDA(coo, XPU, IdType, "COOGetRowNNZ", {
    ret = impl::COOGetRowNNZ<XPU, IdType>(coo, row);
  });
  return ret;
}

std::pair<NDArray, NDArray> COOGetRowDataAndIndices(COOMatrix coo, int64_t row) {
  std::pair<NDArray, NDArray> ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOGetRowDataAndIndices", {
    ret = impl::COOGetRowDataAndIndices<XPU, IdType>(coo, row);
  });
  return ret;
}

std::vector<NDArray> COOGetDataAndIndices(
    COOMatrix coo, NDArray rows, NDArray cols) {
  std::vector<NDArray> ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOGetDataAndIndices", {
    ret = impl::COOGetDataAndIndices<XPU, IdType>(coo, rows, cols);
  });
  return ret;
}

NDArray COOGetData(COOMatrix coo, NDArray rows, NDArray cols) {
  NDArray ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOGetData", {
    ret = impl::COOGetData<XPU, IdType>(coo, rows, cols);
  });
  return ret;
}

COOMatrix COOTranspose(COOMatrix coo) {
  return COOMatrix(coo.num_cols, coo.num_rows, coo.col, coo.row, coo.data);
}

CSRMatrix COOToCSR(COOMatrix coo) {
  CSRMatrix ret;
  ATEN_XPU_SWITCH_CUDA(coo.row->ctx.device_type, XPU, "COOToCSR", {
    ATEN_ID_TYPE_SWITCH(coo.row->dtype, IdType, {
      ret = impl::COOToCSR<XPU, IdType>(coo);
    });
  });
  return ret;
}

COOMatrix COOSliceRows(COOMatrix coo, int64_t start, int64_t end) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOSliceRows", {
    ret = impl::COOSliceRows<XPU, IdType>(coo, start, end);
  });
  return ret;
}

COOMatrix COOSliceRows(COOMatrix coo, NDArray rows) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOSliceRows", {
    ret = impl::COOSliceRows<XPU, IdType>(coo, rows);
  });
  return ret;
}

COOMatrix COOSliceMatrix(COOMatrix coo, NDArray rows, NDArray cols) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOSliceMatrix", {
    ret = impl::COOSliceMatrix<XPU, IdType>(coo, rows, cols);
  });
  return ret;
}

void COOSort_(COOMatrix* mat, bool sort_column) {
  if ((mat->row_sorted && !sort_column) || mat->col_sorted)
    return;
  ATEN_XPU_SWITCH_CUDA(mat->row->ctx.device_type, XPU, "COOSort_", {
    ATEN_ID_TYPE_SWITCH(mat->row->dtype, IdType, {
      impl::COOSort_<XPU, IdType>(mat, sort_column);
    });
  });
}

std::pair<bool, bool> COOIsSorted(COOMatrix coo) {
  if (coo.row->shape[0] <= 1)
    return {true, true};
  std::pair<bool, bool> ret;
  ATEN_COO_SWITCH_CUDA(coo, XPU, IdType, "COOIsSorted", {
    ret = impl::COOIsSorted<XPU, IdType>(coo);
  });
  return ret;
}

COOMatrix COOReorder(COOMatrix coo, runtime::NDArray new_row_ids, runtime::NDArray new_col_ids) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOReorder", {
    ret = impl::COOReorder<XPU, IdType>(coo, new_row_ids, new_col_ids);
  });
  return ret;
}

COOMatrix COORemove(COOMatrix coo, IdArray entries) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COORemove", {
    ret = impl::COORemove<XPU, IdType>(coo, entries);
  });
  return ret;
}

COOMatrix COORowWiseSampling(
    COOMatrix mat, IdArray rows, int64_t num_samples, FloatArray prob, bool replace) {
  COOMatrix ret;
  ATEN_COO_SWITCH(mat, XPU, IdType, "COORowWiseSampling", {
    if (IsNullArray(prob)) {
      ret = impl::COORowWiseSamplingUniform<XPU, IdType>(mat, rows, num_samples, replace);
    } else {
      ATEN_FLOAT_TYPE_SWITCH(prob->dtype, FloatType, "probability", {
        ret = impl::COORowWiseSampling<XPU, IdType, FloatType>(
            mat, rows, num_samples, prob, replace);
      });
    }
  });
  return ret;
}

COOMatrix COORowWiseTopk(
    COOMatrix mat, IdArray rows, int64_t k, FloatArray weight, bool ascending) {
  COOMatrix ret;
  ATEN_COO_SWITCH(mat, XPU, IdType, "COORowWiseTopk", {
    ATEN_DTYPE_SWITCH(weight->dtype, DType, "weight", {
      ret = impl::COORowWiseTopk<XPU, IdType, DType>(
          mat, rows, k, weight, ascending);
    });
  });
  return ret;
}

std::pair<COOMatrix, IdArray> COOCoalesce(COOMatrix coo) {
  std::pair<COOMatrix, IdArray> ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOCoalesce", {
    ret = impl::COOCoalesce<XPU, IdType>(coo);
  });
  return ret;
}

COOMatrix COOLineGraph(const COOMatrix &coo, bool backtracking) {
  COOMatrix ret;
  ATEN_COO_SWITCH(coo, XPU, IdType, "COOLineGraph", {
    ret = impl::COOLineGraph<XPU, IdType>(coo, backtracking);
  });
  return ret;
}

COOMatrix UnionCoo(const std::vector<COOMatrix>& coos) {
  COOMatrix ret;
  CHECK_GT(coos.size(), 1) << "UnionCoo creates a union of multiple COOMatrixes";
  // sanity check
  for (size_t i = 1; i < coos.size(); ++i) {
    CHECK_EQ(coos[0].num_rows, coos[i].num_rows) <<
      "UnionCoo requires both COOMatrix have same number of rows";
    CHECK_EQ(coos[0].num_cols, coos[i].num_cols) <<
      "UnionCoo requires both COOMatrix have same number of cols";
    CHECK_SAME_CONTEXT(coos[0].row, coos[i].row);
    CHECK_SAME_DTYPE(coos[0].row, coos[i].row);
  }

  // we assume the number of coos is not large in common cases
  std::vector<IdArray> coo_row;
  std::vector<IdArray> coo_col;
  bool has_data = false;

  for (size_t i = 0; i < coos.size(); ++i) {
    coo_row.push_back(coos[i].row);
    coo_col.push_back(coos[i].col);
    has_data |= COOHasData(coos[i]);
  }

  IdArray row = Concat(coo_row);
  IdArray col = Concat(coo_col);
  IdArray data = NullArray();

  if (has_data) {
    std::vector<IdArray> eid_data;
    eid_data.push_back(COOHasData(coos[0]) ?
                       coos[0].data :
                       Range(0,
                             coos[0].row->shape[0],
                             coos[0].row->dtype.bits,
                             coos[0].row->ctx));
    int64_t num_edges = coos[0].row->shape[0];
    for (size_t i = 1; i < coos.size(); ++i) {
      eid_data.push_back(COOHasData(coos[i]) ?
                         coos[i].data + num_edges :
                         Range(num_edges,
                               num_edges + coos[i].row->shape[0],
                               coos[i].row->dtype.bits,
                               coos[i].row->ctx));
      num_edges += coos[i].row->shape[0];
    }

    data = Concat(eid_data);
  }

  return COOMatrix(
    coos[0].num_rows,
    coos[0].num_cols,
    row,
    col,
    data,
    false,
    false);
}


std::tuple<COOMatrix, IdArray, IdArray>
COOToSimple(const COOMatrix& coo) {
  // coo column sorted
  const COOMatrix sorted_coo = COOSort(coo, true);
  const IdArray eids_shuffled = COOHasData(sorted_coo) ?
    sorted_coo.data :
    Range(0, sorted_coo.row->shape[0], sorted_coo.row->dtype.bits, sorted_coo.row->ctx);
  const auto &coalesced_result = COOCoalesce(sorted_coo);
  const COOMatrix &coalesced_adj = coalesced_result.first;
  const IdArray &count = coalesced_result.second;

  /*
   * eids_shuffled actually already contains the mapping from old edge space to the
   * new one:
   *
   * * eids_shuffled[0:count[0]] indicates the original edge IDs that coalesced into new
   *   edge #0.
   * * eids_shuffled[count[0]:count[0] + count[1]] indicates those that coalesced into
   *   new edge #1.
   * * eids_shuffled[count[0] + count[1]:count[0] + count[1] + count[2]] indicates those
   *   that coalesced into new edge #2.
   * * etc.
   *
   * Here, we need to translate eids_shuffled to an array "eids_remapped" such that
   * eids_remapped[i] indicates the new edge ID the old edge #i is mapped to.  The
   * translation can simply be achieved by (in numpy code):
   *
   *     new_eid_for_eids_shuffled = np.range(len(count)).repeat(count)
   *     eids_remapped = np.zeros_like(new_eid_for_eids_shuffled)
   *     eids_remapped[eids_shuffled] = new_eid_for_eids_shuffled
   */
  const IdArray new_eids = Range(
    0, coalesced_adj.row->shape[0], coalesced_adj.row->dtype.bits, coalesced_adj.row->ctx);
  const IdArray eids_remapped = Scatter(Repeat(new_eids, count), eids_shuffled);

  COOMatrix ret = COOMatrix(
    coalesced_adj.num_rows,
    coalesced_adj.num_cols,
    coalesced_adj.row,
    coalesced_adj.col,
    NullArray(),
    true,
    true);
  return std::make_tuple(ret, count, eids_remapped);
}

///////////////////////// Graph Traverse routines //////////////////////////
Frontiers BFSNodesFrontiers(const CSRMatrix& csr, IdArray source) {
  Frontiers ret;
  CHECK_EQ(csr.indptr->ctx.device_type, source->ctx.device_type) <<
    "Graph and source should in the same device context";
  CHECK_EQ(csr.indices->dtype, source->dtype) <<
    "Graph and source should in the same dtype";
  CHECK_EQ(csr.num_rows, csr.num_cols) <<
    "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH(source->ctx.device_type, XPU, "BFSNodesFrontiers", {
    ATEN_ID_TYPE_SWITCH(source->dtype, IdType, {
      ret = impl::BFSNodesFrontiers<XPU, IdType>(csr, source);
    });
  });
  return ret;
}

Frontiers BFSEdgesFrontiers(const CSRMatrix& csr, IdArray source) {
  Frontiers ret;
  CHECK_EQ(csr.indptr->ctx.device_type, source->ctx.device_type) <<
    "Graph and source should in the same device context";
  CHECK_EQ(csr.indices->dtype, source->dtype) <<
    "Graph and source should in the same dtype";
  CHECK_EQ(csr.num_rows, csr.num_cols) <<
    "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH(source->ctx.device_type, XPU, "BFSEdgesFrontiers", {
    ATEN_ID_TYPE_SWITCH(source->dtype, IdType, {
      ret = impl::BFSEdgesFrontiers<XPU, IdType>(csr, source);
    });
  });
  return ret;
}

Frontiers TopologicalNodesFrontiers(const CSRMatrix& csr) {
  Frontiers ret;
  CHECK_EQ(csr.num_rows, csr.num_cols) <<
    "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH(csr.indptr->ctx.device_type, XPU, "TopologicalNodesFrontiers", {
    ATEN_ID_TYPE_SWITCH(csr.indices->dtype, IdType, {
      ret = impl::TopologicalNodesFrontiers<XPU, IdType>(csr);
    });
  });
  return ret;
}

Frontiers DGLDFSEdges(const CSRMatrix& csr, IdArray source) {
  Frontiers ret;
  CHECK_EQ(csr.indptr->ctx.device_type, source->ctx.device_type) <<
    "Graph and source should in the same device context";
  CHECK_EQ(csr.indices->dtype, source->dtype) <<
    "Graph and source should in the same dtype";
  CHECK_EQ(csr.num_rows, csr.num_cols) <<
    "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH(source->ctx.device_type, XPU, "DGLDFSEdges", {
    ATEN_ID_TYPE_SWITCH(source->dtype, IdType, {
      ret = impl::DGLDFSEdges<XPU, IdType>(csr, source);
    });
  });
  return ret;
}

Frontiers DGLDFSLabeledEdges(const CSRMatrix& csr,
                             IdArray source,
                             const bool has_reverse_edge,
                             const bool has_nontree_edge,
                             const bool return_labels) {
  Frontiers ret;
  CHECK_EQ(csr.indptr->ctx.device_type, source->ctx.device_type) <<
    "Graph and source should in the same device context";
  CHECK_EQ(csr.indices->dtype, source->dtype) <<
    "Graph and source should in the same dtype";
  CHECK_EQ(csr.num_rows, csr.num_cols) <<
    "Graph traversal can only work on square-shaped CSR.";
  ATEN_XPU_SWITCH(source->ctx.device_type, XPU, "DGLDFSLabeledEdges", {
    ATEN_ID_TYPE_SWITCH(source->dtype, IdType, {
      ret = impl::DGLDFSLabeledEdges<XPU, IdType>(csr,
                                                  source,
                                                  has_reverse_edge,
                                                  has_nontree_edge,
                                                  return_labels);
    });
  });
  return ret;
}


///////////////////////// C APIs /////////////////////////
DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetFormat")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    SparseMatrixRef spmat = args[0];
    *rv = spmat->format;
  });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetNumRows")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    SparseMatrixRef spmat = args[0];
    *rv = spmat->num_rows;
  });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetNumCols")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    SparseMatrixRef spmat = args[0];
    *rv = spmat->num_cols;
  });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetIndices")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    SparseMatrixRef spmat = args[0];
    const int64_t i = args[1];
    *rv = spmat->indices[i];
  });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLSparseMatrixGetFlags")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    SparseMatrixRef spmat = args[0];
    List<Value> flags;
    for (bool flg : spmat->flags) {
      flags.push_back(Value(MakeValue(flg)));
    }
    *rv = flags;
  });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLCreateSparseMatrix")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const int32_t format = args[0];
    const int64_t nrows = args[1];
    const int64_t ncols = args[2];
    const List<Value> indices = args[3];
    const List<Value> flags = args[4];
    std::shared_ptr<SparseMatrix> spmat(new SparseMatrix(
          format, nrows, ncols,
          ListValueToVector<IdArray>(indices),
          ListValueToVector<bool>(flags)));
    *rv = SparseMatrixRef(spmat);
  });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLExistSharedMemArray")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    const std::string name = args[0];
#ifndef _WIN32
    *rv = SharedMemory::Exist(name);
#else
    *rv = false;
#endif  // _WIN32
  });

DGL_REGISTER_GLOBAL("ndarray._CAPI_DGLArrayCastToSigned")
.set_body([] (DGLArgs args, DGLRetValue* rv) {
    NDArray array = args[0];
    CHECK_EQ(array->dtype.code, kDLUInt);
    std::vector<int64_t> shape(array->shape, array->shape + array->ndim);
    DLDataType dtype = array->dtype;
    dtype.code = kDLInt;
    *rv = array.CreateView(shape, dtype, 0);
  });

}  // namespace aten
}  // namespace dgl

std::ostream& operator << (std::ostream& os, dgl::runtime::NDArray array) {
  return os << dgl::aten::ToDebugString(array);
}
