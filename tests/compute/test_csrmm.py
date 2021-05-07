import numpy as np
import scipy.sparse as ssp
import pytest
import dgl
from utils import parametrize_dtype
import backend as F

def _random_simple_graph(idtype, dtype, ctx, M, N, max_nnz, srctype, dsttype, etype):
    src = np.random.randint(0, M, (max_nnz,))
    dst = np.random.randint(0, N, (max_nnz,))
    val = np.random.randn(max_nnz)
    a = ssp.csr_matrix((val, (src, dst)), shape=(M, N))
    a.sum_duplicates()
    a = a.tocoo()
    # shuffle edges
    perm = np.random.permutation(a.nnz)
    row = a.row[perm]
    col = a.col[perm]
    val = a.data[perm]
    a = ssp.csr_matrix((val, (row, col)), shape=(M, N))

    A = dgl.heterograph(
        {(srctype, etype, dsttype): (
            F.copy_to(F.tensor(row, dtype=idtype), ctx),
            F.copy_to(F.tensor(col, dtype=idtype), ctx))},
        num_nodes_dict={srctype: a.shape[0], dsttype: a.shape[1]})
    A.edata['w'] = F.copy_to(F.tensor(val, dtype=dtype), ctx)
    return a, A

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
def test_csrmm(idtype, dtype):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, 9000, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 600, 700, 9000, 'B', 'C', 'BC')
    C, C_weights = dgl.sparse._csrmm(A._graph, A.edata['w'], B._graph, B.edata['w'], 2)
    C_adj = C.adjacency_matrix_scipy(0, True, 'csr')
    C_adj.data = F.asnumpy(C_weights)
    C_adj = F.tensor(C_adj.todense(), dtype=dtype)
    c = F.tensor((a * b).todense(), dtype=dtype)
    assert F.allclose(C_adj, c)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
@pytest.mark.parametrize('num_vtypes', [1, 2])
def test_csrmm_backward(idtype, dtype, num_vtypes):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 4, 3, 6, 'B', 'A' if num_vtypes == 1 else 'C', 'BA')
    A_row, A_col = A.edges(order='eid')
    B_row, B_col = B.edges(order='eid')
    A_row = F.asnumpy(A_row)
    A_col = F.asnumpy(A_col)
    B_row = F.asnumpy(B_row)
    B_col = F.asnumpy(B_col)
    a_dense = F.attach_grad(F.tensor(a.todense(), dtype=dtype))
    b_dense = F.attach_grad(F.tensor(b.todense(), dtype=dtype))

    A.edata['w'] = F.attach_grad(A.edata['w'])
    B.edata['w'] = F.attach_grad(B.edata['w'])

    with F.record_grad():
        C = dgl.adj_product_graph(A, B, 'w')
        assert len(C.ntypes) == num_vtypes
        assert len(C.etypes) == 1
        C_dense = np.zeros((3, 3))
        C_row, C_col = C.edges(order='eid')
        C_row = F.asnumpy(C_row)
        C_col = F.asnumpy(C_col)
        C_dense[C_row, C_col] = F.asnumpy(C.edata['w'])
        c_dense = F.matmul(a_dense, b_dense)
        assert np.allclose(C_dense, F.asnumpy(c_dense), rtol=1e-4, atol=1e-4)

        F.backward(F.reduce_sum(C.edata['w']) + F.reduce_sum(c_dense))
        a_dense_grad = F.asnumpy(F.grad(a_dense))[A_row, A_col]
        b_dense_grad = F.asnumpy(F.grad(b_dense))[B_row, B_col]
        A_spspmm_grad = F.asnumpy(F.grad(A.edata['w']))
        B_spspmm_grad = F.asnumpy(F.grad(B.edata['w']))
        assert np.allclose(a_dense_grad, A_spspmm_grad, rtol=1e-4, atol=1e-4)
        assert np.allclose(b_dense_grad, B_spspmm_grad, rtol=1e-4, atol=1e-4)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
def test_csrsum(idtype, dtype):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, 9000, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, 9000, 'A', 'B', 'AB')
    C, C_weights = dgl.sparse._csrsum([A._graph, B._graph], [A.edata['w'], B.edata['w']])
    C_adj = C.adjacency_matrix_scipy(0, True, 'csr')
    C_adj.data = F.asnumpy(C_weights)
    C_adj = F.tensor(C_adj.todense(), dtype=dtype)
    c = F.tensor((a + b).todense(), dtype=dtype)
    assert F.allclose(C_adj, c)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
@pytest.mark.parametrize('nelems', [1, 2])
def test_csrsum_backward(idtype, dtype, nelems):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    A_row, A_col = A.edges(order='eid')
    B_row, B_col = B.edges(order='eid')
    A_row = F.asnumpy(A_row)
    A_col = F.asnumpy(A_col)
    B_row = F.asnumpy(B_row)
    B_col = F.asnumpy(B_col)
    a_dense = F.attach_grad(F.tensor(a.todense(), dtype=dtype))
    b_dense = F.attach_grad(F.tensor(b.todense(), dtype=dtype))

    A.edata['w'] = F.attach_grad(A.edata['w'])
    B.edata['w'] = F.attach_grad(B.edata['w'])

    with F.record_grad():
        if nelems == 2:
            # Test for two element case
            C = dgl.adj_sum_graph([A, B], 'w')
            assert C.canonical_etypes == A.canonical_etypes
            C_dense = np.zeros((3, 4))
            C_row, C_col = C.edges(order='eid')
            C_row = F.asnumpy(C_row)
            C_col = F.asnumpy(C_col)
            C_dense[C_row, C_col] = F.asnumpy(C.edata['w'])
            c_dense = a_dense + b_dense
            assert np.allclose(C_dense, F.asnumpy(c_dense), rtol=1e-4, atol=1e-4)

            F.backward(F.reduce_sum(C.edata['w']) + F.reduce_sum(c_dense))
            a_dense_grad = F.asnumpy(F.grad(a_dense))[A_row, A_col]
            b_dense_grad = F.asnumpy(F.grad(b_dense))[B_row, B_col]
            A_spspmm_grad = F.asnumpy(F.grad(A.edata['w']))
            B_spspmm_grad = F.asnumpy(F.grad(B.edata['w']))
            assert np.allclose(a_dense_grad, A_spspmm_grad, rtol=1e-4, atol=1e-4)
            assert np.allclose(b_dense_grad, B_spspmm_grad, rtol=1e-4, atol=1e-4)
        elif nelems == 1:
            # Test for single element case
            C = dgl.adj_sum_graph([A], 'w')
            assert C.canonical_etypes == A.canonical_etypes
            C_dense = np.zeros((3, 4))
            C_row, C_col = C.edges(order='eid')
            C_row = F.asnumpy(C_row)
            C_col = F.asnumpy(C_col)
            C_dense[C_row, C_col] = F.asnumpy(C.edata['w'])
            c_dense = a_dense
            assert np.allclose(C_dense, F.asnumpy(c_dense), rtol=1e-4, atol=1e-4)

            F.backward(F.reduce_sum(C.edata['w']) + F.reduce_sum(c_dense))
            a_dense_grad = F.asnumpy(F.grad(a_dense))[A_row, A_col]
            A_spspmm_grad = F.asnumpy(F.grad(A.edata['w']))
            assert np.allclose(a_dense_grad, A_spspmm_grad, rtol=1e-4, atol=1e-4)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
@pytest.mark.parametrize('A_nnz', [9000, 0])
@pytest.mark.parametrize('B_nnz', [9000, 0])
def test_csrmask(idtype, dtype, A_nnz, B_nnz):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, A_nnz, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 500, 600, B_nnz, 'A', 'B', 'AB')
    C = dgl.sparse._csrmask(A._graph, A.edata['w'], B._graph)
    B_row, B_col = B.edges(order='eid')
    B_row = F.asnumpy(B_row)
    B_col = F.asnumpy(B_col)
    c = F.tensor(a.todense()[B_row, B_col], dtype)
    assert F.allclose(C, c)

@parametrize_dtype
@pytest.mark.parametrize('dtype', [F.float32, F.float64])
def test_csrmask_backward(idtype, dtype):
    a, A = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    b, B = _random_simple_graph(idtype, dtype, F.ctx(), 3, 4, 6, 'A', 'B', 'AB')
    A_row, A_col = A.edges(order='eid')
    B_row, B_col = B.edges(order='eid')
    A_row = F.asnumpy(A_row)
    A_col = F.asnumpy(A_col)
    B_row = F.asnumpy(B_row)
    B_col = F.asnumpy(B_col)
    a_dense = F.attach_grad(F.tensor(a.todense(), dtype=dtype))

    A.edata['w'] = F.attach_grad(A.edata['w'])

    with F.record_grad():
        # Test for two element case
        C1 = F.csrmask(A._graph, A.edata['w'], B._graph)
        if dgl.backend.backend_name == 'tensorflow':
            import tensorflow as tf
            C2 = tf.gather_nd(a_dense, tf.stack([B_row, B_col], 1))
        else:
            C2 = a_dense[B_row, B_col]
        assert F.allclose(C1, C2, rtol=1e-4, atol=1e-4)

        F.backward(F.reduce_sum(C1) + F.reduce_sum(C2))
        a_dense_grad = F.asnumpy(F.grad(a_dense))[A_row, A_col]
        A_spspmm_grad = F.asnumpy(F.grad(A.edata['w']))
        assert np.allclose(a_dense_grad, A_spspmm_grad, rtol=1e-4, atol=1e-4)


if __name__ == '__main__':
    test_csrmm(F.int32, F.float32)
    test_csrmm(F.int64, F.float32)
    test_csrsum(F.int32, F.float32)
    test_csrsum(F.int64, F.float32)
    test_csrmask(F.int32, F.float32, 9000, 9000)
    test_csrmask(F.int64, F.float32, 9000, 0)
    test_csrmask(F.int32, F.float32, 0, 9000)
    test_csrmask(F.int64, F.float32, 0, 0)
    test_csrmm_backward(F.int32, F.float32, 1)
    test_csrmm_backward(F.int64, F.float32, 1)
    test_csrmm_backward(F.int32, F.float32, 2)
    test_csrmm_backward(F.int64, F.float32, 2)
    test_csrsum_backward(F.int32, F.float32, 1)
    test_csrsum_backward(F.int64, F.float32, 1)
    test_csrsum_backward(F.int32, F.float32, 2)
    test_csrsum_backward(F.int64, F.float32, 2)
    test_csrmask_backward(F.int32, F.float32)
    test_csrmask_backward(F.int64, F.float32)
