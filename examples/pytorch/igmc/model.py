"""IGMC modules"""

import math 
import torch as th 
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import RelGraphConv

def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)

class IGMC(nn.Module):
    # The GNN model of Inductive Graph-based Matrix Completion. 
    # Use RGCN convolution + center-nodes readout.
    
    def __init__(self, in_feats, gconv=RelGraphConv, latent_dim=[32, 32, 32, 32], 
                num_relations=5, num_bases=2, regression=False, edge_dropout=0.2, 
                force_undirected=False, side_features=False, n_side_features=0, 
                multiply_by=1):
        super(IGMC, self).__init__()

        self.regression = regression
        self.edge_dropout = edge_dropout
        self.force_undirected = force_undirected
        self.side_features = side_features
        self.multiply_by = multiply_by

        self.convs = th.nn.ModuleList()
        self.convs.append(gconv(in_feats, latent_dim[0], num_relations, 
                                num_bases=num_bases, self_loop=True, low_mem=True))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(gconv(latent_dim[i], latent_dim[i+1], num_relations, 
                                    num_bases=num_bases, self_loop=True, low_mem=True))
        
        self.lin1 = nn.Linear(2 * sum(latent_dim), 128)
        if side_features:
            self.lin1 = nn.Linear(2 * sum(latent_dim) + n_side_features, 128)
        if self.regression:
            self.lin2 = nn.Linear(128, 1)
        else:
            assert False
            # self.lin2 = nn.Linear(128, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.convs:
            size = conv.num_bases * conv.in_feat
            uniform(size, conv.weight)
            uniform(size, conv.w_comp)
            uniform(size, conv.loop_weight)
            uniform(size, conv.h_bias)
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    # @profile
    def forward(self, block):
        block = edge_drop(block, self.edge_dropout, self.training)

        concat_states = []
        x = block.ndata['nlabel'] # th.cat([subgraph.ndata['nlabel'], subgraph.ndata['refex']], dim=1)
        for conv in self.convs:
            # edge mask zero denotes the edge dropped
            x = th.tanh(conv(block, x, block.edata['etype'], 
                             norm=block.edata['edge_mask'].unsqueeze(1)))
            concat_states.append(x)
        concat_states = th.cat(concat_states, 1)
        
        users = block.ndata['nlabel'][:, 0] == 1
        items = block.ndata['nlabel'][:, 1] == 1
        x = th.cat([concat_states[users], concat_states[items]], 1)
        # if self.side_features:
        #     x = th.cat([x, data.u_feature, data.v_feature], 1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        if self.regression:
            return x[:, 0] * self.multiply_by
        else:
            assert False
            # return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__

def edge_drop(graph, edge_dropout=0.2, training=True):
    assert edge_dropout >= 0.0 and edge_dropout <= 1.0, 'Invalid dropout rate.'

    if not training:
        return graph

    # set edge mask to zero in directional mode
    src, _ = graph.edges()
    to_drop = src.new_full((graph.number_of_edges(), ), edge_dropout, dtype=th.float)
    to_drop = th.bernoulli(to_drop).to(th.bool)
    graph.edata['edge_mask'][to_drop] = 0

    return graph
