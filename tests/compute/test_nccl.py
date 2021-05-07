from dgl.cuda import nccl
import unittest
import backend as F


def gen_test_id():
    return '{:0256x}'.format(78236728318467363)

@unittest.skipIf(F._default_context_str == 'cpu', reason="NCCL only runs on GPU.")
def test_nccl_id():
    nccl_id = nccl.UniqueId()
    text = str(nccl_id)
    nccl_id2 = nccl.UniqueId(id_str=text)

    assert nccl_id == nccl_id2

    nccl_id2 = nccl.UniqueId(gen_test_id())

    assert nccl_id2 != nccl_id

    nccl_id3 = nccl.UniqueId(str(nccl_id2))

    assert nccl_id2 == nccl_id3


@unittest.skipIf(F._default_context_str == 'cpu', reason="NCCL only runs on GPU.")
def test_nccl_sparse_push_single():
    nccl_id = nccl.UniqueId()
    comm = nccl.Communicator(1, 0, nccl_id)

    index = F.randint([10000], F.int32, F.ctx(), 0, 10000)
    value = F.uniform([10000, 100], F.float32, F.ctx(), -1.0, 1.0)

    ri, rv = comm.sparse_all_to_all_push(index, value, 'remainder')
    assert F.array_equal(ri, index)
    assert F.array_equal(rv, value)

@unittest.skipIf(F._default_context_str == 'cpu', reason="NCCL only runs on GPU.")
def test_nccl_sparse_pull_single():
    nccl_id = nccl.UniqueId()
    comm = nccl.Communicator(1, 0, nccl_id)

    req_index = F.randint([10000], F.int64, F.ctx(), 0, 100000)
    value = F.uniform([100000, 100], F.float32, F.ctx(), -1.0, 1.0)

    rv = comm.sparse_all_to_all_pull(req_index, value, 'remainder')
    exp_rv = F.gather_row(value, req_index)
    assert F.array_equal(rv, exp_rv)


if __name__ == '__main__':
    test_nccl_id()
    test_nccl_sparse_push_single()
    test_nccl_sparse_pull_single()
