import dgl
import torch
import numba
from numba.core import types
from numba.typed import Dict
import numpy as np

@numba.njit
def find_indices_in(a, b):
    d = Dict.empty(key_type=types.int64, value_type=types.int64)
    for i, v in enumerate(b):
        d[v] = i
    ai = np.zeros_like(a)
    for i, v in enumerate(a):
        ai[i] = d.get(v, -1)
    return ai

@numba.jit
def union(*arrays):
    # Faster than np.union1d and torch.unique(torch.cat(...))
    s = set()
    for a in arrays:
        s.update(a)
    a = np.asarray(list(s))
    return a


class LADIESNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, nodes_per_layer, weight='w', out_weight='w', replace=False):
        super().__init__(len(nodes_per_layer), return_eids=False)
        self.nodes_per_layer = nodes_per_layer
        self.edge_weight = weight
        self.output_weight = out_weight
        self.replace = replace

    
    def compute_prob(self, g, seed_nodes, weight, exclude_eids):
        """
        g : the whole graph
        seed_nodes : the output nodes for the current layer
        weight : the weight of the edges

        return : the unnormalized probability of the candidate nodes, as well as the subgraph
                 containing all the edges from the candidate nodes to the output nodes.
        """
        insg = dgl.in_subgraph(g, seed_nodes)
        insg = self._exclude_eids(insg, exclude_eids)
        insg = dgl.compact_graphs(insg, seed_nodes)
        out_frontier = dgl.reverse(insg, copy_edata=True)
        weight = weight[out_frontier.edata[dgl.EID]]
        prob = dgl.ops.copy_e_sum(out_frontier, weight ** 2)
        return prob, insg

    
    def select_neighbors(self, seed_nodes, cand_nodes, prob, num, replace):
        """
        seed_nodes : output nodes
        cand_nodes : candidate nodes.  Must contain all output nodes in @seed_nodes
        prob : unnormalized probability of each candidate node
        num : number of neighbors to sample

        return : the set of input nodes in terms of their indices in @cand_nodes, and also the indices of
                 seed nodes in the selected nodes.
        """
        # The returned nodes should be a union of seed_nodes plus @num nodes from cand_nodes.
        # Because compute_prob returns a compacted subgraph and a list of probabilities,
        # we need to find the corresponding local IDs of the resulting union in the subgraph
        # so that we can compute the edge weights of the block.
        # This is why we need a find_indices_in() function.
        neighbor_nodes_idx = torch.multinomial(prob, num, replacement=replace).cpu().numpy()
        seed_nodes_idx = find_indices_in(seed_nodes.cpu().numpy(), cand_nodes.cpu().numpy())
        assert seed_nodes_idx.min() != -1
        neighbor_nodes_idx = union(neighbor_nodes_idx, seed_nodes_idx)
        seed_nodes_local_idx = torch.from_numpy(find_indices_in(seed_nodes_idx, neighbor_nodes_idx))
        assert seed_nodes_idx.min().item() != -1
        neighbor_nodes_idx = torch.from_numpy(neighbor_nodes_idx)
        return neighbor_nodes_idx, seed_nodes_local_idx

    
    def generate_block(self, insg, neighbor_nodes_idx, seed_nodes_local_idx, P_sg, W_sg, weight_name,
                       return_eids):
        """
        insg : the subgraph yielded by compute_prob()
        neighbor_nodes_idx : the sampled nodes from the subgraph @insg, yielded by select_neighbors()
        seed_nodes_local_idx : the indices of seed nodes in the selected neighbor nodes, also yielded
                               by select_neighbors()
        P_sg : unnormalized probability of each node being sampled, yielded by compute_prob()
        W_sg : edge weights of @insg

        return : the block.
        """
        sg = insg.subgraph(neighbor_nodes_idx)
        nids = insg.ndata[dgl.NID][sg.ndata[dgl.NID]]
        P = P_sg[neighbor_nodes_idx]
        W = W_sg[sg.edata[dgl.EID]]
        W_tilde = dgl.ops.e_div_u(sg, W, P)
        W_tilde_sum = dgl.ops.copy_e_sum(sg, W_tilde)
        W_tilde = dgl.ops.e_div_v(sg, W_tilde, W_tilde_sum)

        block = dgl.to_block(sg, seed_nodes_local_idx)
        block.edata[weight_name] = W_tilde
        # correct node ID mapping
        block.srcdata[dgl.NID] = nids[block.srcdata[dgl.NID]]
        block.dstdata[dgl.NID] = nids[block.dstdata[dgl.NID]]

        if return_eids:
            sg_eids = insg.edata[dgl.EID][sg.edata[dgl.EID]]
            block.edata[dgl.EID] = sg_eids[block.edata[dgl.EID]]
        return block

    
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        blocks = []
        exclude_eids = self._convert_exclude_eids(exclude_eids)
        for block_id in reversed(range(self.num_layers)):
            num_nodes_to_sample = self.nodes_per_layer[block_id]
            W = g.edata[self.edge_weight]
            prob, insg = self.compute_prob(g, seed_nodes, W, exclude_eids)
            cand_nodes = insg.ndata[dgl.NID]
            neighbor_nodes_idx, seed_nodes_local_idx = self.select_neighbors(
                seed_nodes, cand_nodes, prob, num_nodes_to_sample, self.replace)
            block = self.generate_block(
                insg, neighbor_nodes_idx, seed_nodes_local_idx, prob,
                W[insg.edata[dgl.EID]], self.output_weight, self.return_eids)
            seed_nodes = block.srcdata[dgl.NID]
            block.create_formats_()
            blocks.insert(0, block)
        return blocks


    def sample_subgraph(self, g, seed_nodes, exclude_eids=None):
        exclude_eids = self._convert_exclude_eids(exclude_eids)
        for layer_id in reversed(range(self.num_layers)):
            num_nodes_to_sample = self.nodes_per_layer[layer_id]
            W = g.edata[self.edge_weight]
            prob, insg = self.compute_prob(g, seed_nodes, W, exclude_eids)
            cand_nodes = insg.ndata[dgl.NID]
            neighbor_nodes_idx, _ = self.select_neighbors(
                seed_nodes, cand_nodes, prob, num_nodes_to_sample, self.replace)
            seed_nodes = cand_nodes[neighbor_nodes_idx]

        subgraph = g.subgraph(seed_nodes)
        if exclude_eids is not None:
            subgraph = dgl.remove_edges(g, exclude_eids)
        return subgraph
