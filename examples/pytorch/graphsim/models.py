import dgl
import torch
import torch.nn as nn
from torch.nn import functional as F
import dgl.nn as dglnn
import dgl.function as fn
import copy
from functools import partial


class MLP(nn.Module):
    def __init__(self, in_feats, out_feats, num_layers=2, hidden=128):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()
        layer = nn.Linear(hidden, out_feats)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(nn.Linear(in_feats, hidden))
        if num_layers > 2:
            for i in range(1, num_layers-1):
                layer = nn.Linear(hidden, hidden)
                nn.init.normal_(layer.weight, std=0.1)
                nn.init.zeros_(layer.bias)
                self.layers.append(layer)
        layer = nn.Linear(hidden, out_feats)
        nn.init.normal_(layer.weight, std=0.1)
        nn.init.zeros_(layer.bias)
        self.layers.append(layer)

    def forward(self, x):
        for l in range(len(self.layers)-1):
            x = self.layers[l](x)
            x = F.relu(x)
        x = self.layers[-1](x)
        return x


class PrepareLayer(nn.Module):
    '''
    Generate edge feature for the model input preparation:
    as well as do the normalization work.
    Parameters
    ==========
    node_feats : int
        Number of node features

    stat : dict
        dictionary which represent the statistics needed for normalization
    '''

    def __init__(self, node_feats, stat):
        super(PrepareLayer, self).__init__()
        self.node_feats = node_feats
        # stat {'median':median,'max':max,'min':min}
        self.stat = stat

    def normalize_input(self, node_feature):
        return (node_feature-self.stat['median'])*(2/(self.stat['max']-self.stat['min']))

    def forward(self, g, node_feature):
        with g.local_scope():
            node_feature = self.normalize_input(node_feature)
            g.ndata['feat'] = node_feature  # Only dynamic feature
            g.apply_edges(fn.u_sub_v('feat', 'feat', 'e'))
            edge_feature = g.edata['e']
            return node_feature, edge_feature


class InteractionNet(nn.Module):
    '''
    Simple Interaction Network
    One Layer interaction network for stellar multi-body problem simulation,
    it has the ability to simulate number of body motion no more than 12
    Parameters
    ==========
    node_feats : int
        Number of node features

    stat : dict
        Statistcics for Denormalization
    '''

    def __init__(self, node_feats, stat):
        super(InteractionNet, self).__init__()
        self.node_feats = node_feats
        self.stat = stat
        edge_fn = partial(MLP, num_layers=5, hidden=150)
        node_fn = partial(MLP, num_layers=2, hidden=100)

        self.in_layer = InteractionLayer(node_feats-3,  # Use velocity only
                                         node_feats,
                                         out_node_feats=2,
                                         out_edge_feats=50,
                                         edge_fn=edge_fn,
                                         node_fn=node_fn,
                                         mode='n_n')

    # Denormalize Velocity only
    def denormalize_output(self, out):
        return out*(self.stat['max'][3:5]-self.stat['min'][3:5])/2+self.stat['median'][3:5]

    def forward(self, g, n_feat, e_feat, global_feats, relation_feats):
        with g.local_scope():
            out_n, out_e = self.in_layer(
                g, n_feat, e_feat, global_feats, relation_feats)
            out_n = self.denormalize_output(out_n)
            return out_n, out_e


class InteractionLayer(nn.Module):
    '''
    Implementation of single layer of interaction network
    Parameters
    ==========
    in_node_feats : int
        Number of node features

    in_edge_feats : int
        Number of edge features

    out_node_feats : int
        Number of node feature after one interaction

    out_edge_feats : int
        Number of edge features after one interaction

    global_feats : int
        Number of global features used as input

    relate_feats : int
        Feature related to the relation between object themselves

    edge_fn : torch.nn.Module
        Function to update edge feature in message generation

    node_fn : torch.nn.Module
        Function to update node feature in message aggregation

    mode : str
        Type of message should the edge carry 
        nne : [src_feat,dst_feat,edge_feat] node feature concat edge feature.
        n_n : [src_feat-edge_feat] node feature subtract from each other.
    '''

    def __init__(self, in_node_feats,
                 in_edge_feats,
                 out_node_feats,
                 out_edge_feats,
                 global_feats=1,
                 relate_feats=1,
                 edge_fn=nn.Linear,
                 node_fn=nn.Linear,
                 mode='nne'):  # 'n_n'
        super(InteractionLayer, self).__init__()
        self.in_node_feats = in_node_feats
        self.in_edge_feats = in_edge_feats
        self.out_edge_feats = out_edge_feats
        self.out_node_feats = out_node_feats
        self.mode = mode
        # MLP for message passing
        input_shape = 2*self.in_node_feats + \
            self.in_edge_feats if mode == 'nne' else self.in_edge_feats+relate_feats
        self.edge_fn = edge_fn(input_shape,
                               self.out_edge_feats)  # 50 in IN paper

        self.node_fn = node_fn(self.in_node_feats+self.out_edge_feats+global_feats,
                               self.out_node_feats)

    # Should be done by apply edge
    def update_edge_fn(self, edges):
        x = torch.cat([edges.src['feat'], edges.dst['feat'],
                       edges.data['feat']], dim=1)
        ret = F.relu(self.edge_fn(
            x)) if self.mode == 'nne' else self.edge_fn(x)
        return {'e': ret}

    # Assume agg comes from build in reduce
    def update_node_fn(self, nodes):
        x = torch.cat([nodes.data['feat'], nodes.data['agg']], dim=1)
        ret = F.relu(self.node_fn(
            x)) if self.mode == 'nne' else self.node_fn(x)
        return {'n': ret}

    def forward(self, g, node_feats, edge_feats, global_feats, relation_feats):
        # print(node_feats.shape,global_feats.shape)
        g.ndata['feat'] = torch.cat([node_feats, global_feats], dim=1)
        g.edata['feat'] = torch.cat([edge_feats, relation_feats], dim=1)
        if self.mode == 'nne':
            g.apply_edges(self.update_edge_fn)
        else:
            g.edata['e'] = self.edge_fn(g.edata['feat'])

        g.update_all(fn.copy_e('e', 'msg'),
                     fn.sum('msg', 'agg'),
                     self.update_node_fn)
        return g.ndata['n'], g.edata['e']
