import torch
import dgl
import dgl.function as fn
from collections import Counter
import numpy as np

def compute_prob(g, seed_nodes, weight):
    out_frontier = dgl.reverse(dgl.in_subgraph(g, seed_nodes), copy_edata=True)
    if out_frontier.number_of_edges() == 0:
        return torch.zeros(g.number_of_nodes(), device=g.device), torch.zeros(0, device=g.device)

    if weight is None:
        edge_weight = torch.ones(out_frontier.number_of_edges(), device=out_frontier.device)
    else:
        edge_weight = out_frontier.edata[weight]
    with out_frontier.local_scope():
        # Sample neighbors on the previous layer
        out_frontier.edata['w'] = edge_weight
        out_frontier.edata['w'] = out_frontier.edata['w'] ** 2
        out_frontier.update_all(fn.copy_e('w', 'm'), fn.sum('m', 'prob'))
        prob = out_frontier.ndata['prob']
        return prob

class LADIESNeighborSampler(dgl.dataloading.BlockSampler):
    def __init__(self, nodes_per_layer, weight=None, out_weight=None, replace=False):
        super().__init__(len(nodes_per_layer), return_eids=True)
        self.nodes_per_layer = nodes_per_layer
        self.weight = weight
        self.replace = replace
        self.out_weight = out_weight

    def sample_frontier(self, block_id, g, seed_nodes):
        num_nodes = self.nodes_per_layer[block_id]
        prob = compute_prob(g, seed_nodes, self.weight)
        candidate_nodes = torch.nonzero(prob, as_tuple=True)[0]

        if not self.replace and len(candidate_nodes) < num_nodes:
            neighbor_nodes = candidate_nodes
        else:
            neighbor_nodes = torch.multinomial(
                prob, self.nodes_per_layer[block_id], replacement=self.replace)
        neighbor_nodes = torch.cat([seed_nodes, neighbor_nodes])
        neighbor_nodes = torch.unique(neighbor_nodes)

        neighbor_graph = dgl.in_subgraph(g, seed_nodes)
        neighbor_graph = dgl.out_subgraph(neighbor_graph, neighbor_nodes)

        # Compute output edge weight
        if self.out_weight is not None:
            with neighbor_graph.local_scope():
                if self.weight is not None:
                    neighbor_graph.edata['P'] = neighbor_graph.edata[self.weight]
                else:
                    neighbor_graph.edata['P'] = torch.ones(neighbor_graph.number_of_edges(), device=neighbor_graph.device)
                neighbor_graph.ndata['S'] = prob
                neighbor_graph.apply_edges(dgl.function.e_div_u('P', 'S', 'P_tilde'))
                # Row normalize
                neighbor_graph.update_all(
                    dgl.function.copy_e('P_tilde', 'P_tilde'),
                    dgl.function.sum('P_tilde', 'P_tilde_sum'))
                neighbor_graph.apply_edges(dgl.function.e_div_v('P_tilde', 'P_tilde_sum', 'P_tilde'))
                w = neighbor_graph.edata['P_tilde']
            neighbor_graph.edata[self.out_weight] = w

        return neighbor_graph
