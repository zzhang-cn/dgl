import torch
import dgl
import dgl.function as fn
from collections import Counter
import numpy as np
import scipy.sparse as ssp
import numba
from numba.core import types
from numba.typed import Dict
import time

def row_normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = ssp.diags(r_inv)

    mx = r_mat_inv.dot(mx)
    return mx

import numpy as np
@numba.njit
def sumbykey(keys, values):
    d = Dict.empty(key_type=types.int64, value_type=types.float32)
    for k, v in zip(keys, values):
        if k not in d:
            d[k] = v
        else:
            d[k] += v
    k = np.zeros((len(d),), np.int64)
    v = np.zeros((len(d),), np.float32)
    for i, (dk, dv) in enumerate(d.items()):
        k[i] = dk
        v[i] = dv
    return k, v

@profile
def sum_rows(csr):
    indices = csr.indices
    data = csr.data
    new_indices, new_data = sumbykey(indices, data)
    return ssp.csr_matrix((new_data, new_indices, np.array([0, len(new_indices)])), shape=(1, csr.shape[1]))

class LADIESNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, g, nodes_per_layer, weight=None, out_weight=None, replace=False):
        super().__init__(len(nodes_per_layer), return_eids=True)
        self.nodes_per_layer = nodes_per_layer
        self.weight = weight
        self.replace = replace
        self.out_weight = out_weight
        self.g = g
        rows, cols = g.edges()
        self.lap_matrix = ssp.csr_matrix((
            self.g.edata[weight].cpu().numpy() if weight is not None else
            np.ones(self.g.num_edges(), dtype='float32'),
            (cols.cpu().numpy(), rows.cpu().numpy())),
            shape=(g.num_nodes(), g.num_nodes()))
        self.eid_matrix = ssp.csr_matrix((
            np.arange(self.g.num_edges()),
            (cols.cpu().numpy(), rows.cpu().numpy())),
            shape=(g.num_nodes(), g.num_nodes()))
        self.T = 0

    @profile
    def sample_frontier(self, block_id, g, seed_nodes):
        t0 = time.time()
        seed_nodes = seed_nodes.cpu().numpy()
        U = self.lap_matrix[seed_nodes, :]

        U_sqr = U.multiply(U)
        p = sum_rows(U_sqr.tocsr())
        p.data /= p.data.sum()

        s_num = min(p.nnz, self.nodes_per_layer[block_id])
        p_indices = p.indices
        p_data = p.data

        after_nodes = np.random.choice(p_indices, s_num, p=p_data, replace=self.replace)
        after_nodes = np.unique(np.concatenate([after_nodes, seed_nodes]))

        p.data = 1 / p.data
        adj = U[:, after_nodes].multiply(p[:, after_nodes])
        adj = row_normalize(adj)
        adj_data = adj.data
        tt = time.time()
        self.T += tt - t0

        eids = self.eid_matrix[seed_nodes, :]
        eids = eids[:, after_nodes]
        eid_data = eids.data

        # To copy edge features and node features I use edge_subgraph.
        frontier = g.edge_subgraph(eid_data, preserve_nodes=True)
        frontier.edata[self.out_weight] = torch.from_numpy(adj_data)
        return frontier
