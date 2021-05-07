import dgl
import dgl.function as fn
from collections import Counter
import numpy as np
import scipy.sparse as ssp
import itertools
import backend as F
import networkx as nx
import unittest, pytest
from dgl import DGLError
from utils import parametrize_dtype

def create_test_heterograph(num_nodes, num_adj, num_tags, index_dtype):
    if isinstance(num_adj, int):
        num_adj = [num_adj, num_adj+1]
    num_adj_list = list(np.random.choice(np.arange(num_adj[0], num_adj[1]), num_nodes))
    src = np.concatenate([[i] * num_adj_list[i] for i in range(num_nodes)])
    dst = [np.random.choice(num_nodes, nadj, replace=False) for nadj in num_adj_list]
    dst = np.concatenate(dst)
    return dgl.graph((src, dst), index_dtype=index_dtype)

def check_sort(spm, tag_arr=None, tag_pos=None):
    if tag_arr is None:
        tag_arr = np.arange(spm.shape[0])
    else:
        tag_arr = F.zerocopy_to_numpy(tag_arr)
    if tag_pos is not None:
        tag_pos = F.zerocopy_to_numpy(tag_pos)
    for i in range(spm.shape[0]):
        row = spm.getrow(i)
        dst = row.nonzero()[1]
        if tag_pos is not None:
            tag_pos_row = np.concatenate([[0], tag_pos[i], [len(dst)]])
            tag_pos_ptr = tag_arr[dst[0]]
        for j in range(len(dst) - 1):
            if tag_pos is not None and tag_arr[dst[j]] != tag_pos_ptr:
                return False
            if tag_arr[dst[j]] > tag_arr[dst[j+1]]:
                return False
            if tag_pos is not None and tag_arr[dst[j]] < tag_arr[dst[j+1]]:
                if j+1 != int(tag_pos_row[tag_pos_ptr+1]):
                    return False
                tag_pos_ptr = tag_arr[dst[j+1]]
    return True

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sorting by tag not implemented")
@parametrize_dtype
def test_sort_outplace(idtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, idtype=idtype)
    g.ndata['tag'] = F.tensor(np.random.choice(num_tags, g.number_of_nodes()))
    
    new_g = dgl.sort_out_edges(g, 'tag')
    old_csr = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    new_csr = new_g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(new_csr, new_g.ndata['tag'], new_g.ndata["_TAG_POS"]))
    assert(not check_sort(old_csr, g.ndata['tag']))

    new_g = dgl.sort_in_edges(g, 'tag')
    old_csc = g.adjacency_matrix(scipy_fmt='csr')
    new_csc = new_g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(new_csc, new_g.ndata['tag'], new_g.ndata["_TAG_POS"]))
    assert(not check_sort(old_csc, g.ndata['tag']))

@unittest.skipIf(F._default_context_str == 'gpu', reason="GPU sorting by tag not implemented")
@parametrize_dtype
def test_sort_outplace_bipartite(idtype):
    num_nodes, num_adj, num_tags = 200, [20, 40], 5
    g = create_test_heterograph(num_nodes, num_adj, num_tags, idtype=idtype)
    g = dgl.bipartite(g.edges(), idtype=idtype)
    utag = np.random.choice(num_tags, g.number_of_nodes('_U'))
    vtag = np.random.choice(num_tags, g.number_of_nodes('_U'))

    g.nodes['_V'].data['tag'] = F.tensor(vtag)
    g.nodes['_U'].data['tag'] = F.tensor(utag)

    new_g = dgl.sort_out_edges(g, 'tag')
    old_csr = g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    new_csr = new_g.adjacency_matrix(transpose=True, scipy_fmt='csr')
    assert(check_sort(new_csr, new_g.nodes['_V'].data['tag'], new_g.nodes['_U'].data['_TAG_POS']))
    assert(not check_sort(old_csr, g.nodes['_V'].data['tag']))

    new_g = dgl.sort_in_edges(g, 'tag')
    old_csc = g.adjacency_matrix(scipy_fmt='csr')
    new_csc = new_g.adjacency_matrix(scipy_fmt='csr')
    assert(check_sort(new_csc, new_g.nodes['_U'].data['tag'], new_g.nodes['_V'].data['_TAG_POS']))
    assert(not check_sort(old_csc, g.nodes['_U'].data['tag']))

if __name__ == "__main__":
    test_sort_outplace("int32")
    test_sort_outplace_bipartite("int32")
