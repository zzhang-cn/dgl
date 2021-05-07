
import torch as th
import dgl

import torch.nn as nn
import dgl.function as dglfn
import dgl.nn as dglnn
from dgl import DGLError
from utils import ccorr


class CompGraphConv(nn.Module):
    """
    One layer of composition graph convolutional network, including the 3 major computation forumual:
    ::math::
        1. Formula 1: node feature update
        $$h_v^{k+1}=f(\displaystyle\sum_{(u, r)\in \mathcal{N}(v)}W_{\lambda(r)}^k\phi(h_u^k, h_r^k))$$

        The $W_{\lambda(r)}$ has 3 types, $W_I, W_O, W_S$. Here, the aggregation is performed across
        relations.

        2. Formula 2: edge feature update
        $$h_r^{k+1}=W_{\text{rel}}^k h_r^k$$

        3. Formula 3: layer 0 of edge feature
        $$h_r^0 = W_{\text{rel}}^0z_r$$

        4. Formula 4: input layer of edge
        $$z_r=\displaystyle\sum_{b=1}^{\mathcal{B}}\alpha_{br}v_b$$

        - Mimic Heterograph to form a local module.
     """

    def __init__(self,
                 hid_dim,
                 comp_fn='sub',
                 activation=None,
                 bias=True,
                 dropout=0.0,
                 batchnorm=False):
        super(CompGraphConv, self).__init__()
        self.hid_dim = hid_dim
        self.comp_fn = comp_fn
        self.activation = activation
        self.bias = bias
        self.dropout = nn.Dropout(dropout)
        self.batchnorm = batchnorm

        # define weights of 3 node matrics
        self.W_I = nn.Parameter(th.Tensor(self.hid_dim, self.hid_dim))
        self.W_O = nn.Parameter(th.Tensor(self.hid_dim, self.hid_dim))
        self.W_S = nn.Parameter(th.Tensor(self.hid_dim, self.hid_dim))

        if self.bias:
            self.b_I = nn.Parameter(th.Tensor(self.hid_dim))
            self.b_O = nn.Parameter(th.Tensor(self.hid_dim))
            self.b_S = nn.Parameter(th.Tensor(self.hid_dim))

        # define weights of >0 edge matric
        self.W_rel = nn.Parameter(th.Tensor(self.hid_dim, self.hid_dim))
        self.b_rel = nn.Parameter(th.Tensor(self.hid_dim))

        # initialize all weights
        self._initialize()

    def _initialize(self):
        r"""
        Initialize all weights
        """
        nn.init.xavier_uniform_(self.W_I, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_O, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_S, gain=nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.W_rel, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.b_I)
        nn.init.zeros_(self.b_O)
        nn.init.zeros_(self.b_S)
        nn.init.zeros_(self.b_rel)

    def forward(self, g, n_in_feats, e_in_feats):
        """
        Input graph g should include reversed edges, and at the end of
        forward add self loop to each node

        Notes
        -----
        1. Node feature should have the same dimensions as edges so that
        composition function could apply to them.
        2. In CompGraphConv, the module assume the input graph has been
        added reversed edge for each original edge. For example, if there
        is a ('u' 'e' 'v') edge in the original graph, there must be a
        ('v' 'e_inv' 'u') edge, where e_inv is mannual added to the original
        graph AND edges in this type must have _inv in the end of edge name.

        Parameters
        ----------
        g: DGL Graph
            A DGL heterograph.

        n_in_feats: dict[str, Tensor]
            A dictionary of node type and node features.

        e_in_feats: dict[str, Tensor]
            A dictionary of edge type and edge features.

        """
        with g.local_scope():

            # handle each edge type relation sub_graph
            n_outputs = {ntype: [] for ntype in g.ntypes}

            for stype, etype, dtype in g.canonical_etypes:

                if etype not in e_in_feats:
                    continue

                # extract a subgraph in one edge type
                rel_graph = g[stype, etype, dtype]

                # check edge numbers
                if rel_graph.number_of_edges() == 0:
                    continue

                # assign node and edge features to the subgraph in the name of 'h'
                if stype not in n_in_feats or dtype not in n_in_feats:
                    continue
                else:
                    rel_graph.nodes[stype].data['h'] = n_in_feats[stype]
                    rel_graph.nodes[dtype].data['h'] = n_in_feats[dtype]

                    rel_graph.edges[etype].data['h'] = e_in_feats[etype]

                # compute norm values
                in_norm = rel_graph.in_degrees().float().clamp(min=1)
                in_norm = th.pow(in_norm, -0.5)

                # compute composition function
                if self.comp_fn == 'sub':
                    rel_graph.update_all(dglfn.u_sub_e(lhs_field='h', rhs_field='h', out='m'),
                                         dglfn.sum(msg='m', out='comp_h'))
                elif self.comp_fn == 'mul':
                    rel_graph.update_all(dglfn.u_mul_e(lhs_field='h', rhs_field='h', out='m'),
                                         dglfn.sum(msg='m', out='comp_h'))
                elif self.comp_fn == 'corr':
                    rel_graph.update_all(lambda edges: {'corr_h': ccorr(edges.src['h'], edges.data['h'])},
                                     dglfn.sum(msg='corr_h', out='comp_h'))
                else:
                    raise DGLError('Only supports sub, mul, and ccorr')

                # normalize composition results
                rel_graph.dstdata['comp_h'] = rel_graph.dstdata['comp_h'] * in_norm.view(-1, 1)

                # linear transformation by different weights and add bias values
                if str.endswith(etype, '_inv'):
                    rel_graph.dstdata['comp_h'] = th.matmul(rel_graph.dstdata['comp_h'], self.W_O)
                    if self.bias:
                        rel_graph.dstdata['comp_h'] = rel_graph.dstdata['comp_h'] + self.b_O
                else:
                    rel_graph.dstdata['comp_h'] = th.matmul(rel_graph.dstdata['comp_h'], self.W_I)
                    if self.bias:
                        rel_graph.dstdata['comp_h'] = rel_graph.dstdata['comp_h'] + self.b_I

                # add composition results to each node's list
                n_outputs[dtype].append(rel_graph.dstdata['comp_h'])

            # self loop
            self_rsts = {}
            for ntype in g.ntypes:
                self_rsts[ntype] = th.matmul(g.nodes[ntype].data['h'], self.W_S)
                if self.bias:
                    self_rsts[ntype] = self_rsts[ntype] + self.b_S


        # aggregate all relations' composition results
        n_out_feats = {}
        for ntype, alist in n_outputs.items():
            if len(alist) != 0:
                stacked = th.stack(alist, dim=0)
                n_out_feats[ntype] = th.sum(stacked, dim=0)

                # add self loop results and dropout
                n_out_feats[ntype] = self.dropout(n_out_feats[ntype]) * 2/3 + \
                                     self.dropout(self_rsts.get(ntype)) * 1/3

                # add batchnorm
                if self.batchnorm:
                    bn = nn.BatchNorm1d(self.hid_dim)
                    n_out_feats[ntype] = bn(n_out_feats[ntype])

                # use activate to compute non-linear
                if self.activation is not None:
                    n_out_feats[ntype] = self.activation(n_out_feats[ntype])

        # compute edge embedding of the next layer
        e_out_feats = {}
        for etype, feat in e_in_feats.items():
            e_out_feats[etype] = th.matmul(e_in_feats[etype], self.W_rel) + self.b_rel

            # use activate to compute non-linear
            if self.activation is not None:
                e_out_feats[etype] = self.activation(e_out_feats[etype])

        return n_out_feats, e_out_feats


class CompGCN(nn.Module):
    """
    1. Will use dgl.nn.WeightBasis module to create basis vector of edge；
    2.

    """
    def __init__(self,
                 in_feat_dict,
                 hid_dim,
                 num_layers,
                 out_feat,
                 num_basis,
                 num_rel,
                 comp_fn='sub',
                 dropout=0.0,
                 activation=None,
                 batchnorm=False
                 ):
        super(CompGCN, self).__init__()
        self.in_feat_dict = in_feat_dict
        self.rel_emb_dim = hid_dim
        self.num_layer = num_layers
        self.out_feat = out_feat
        self.num_basis = num_basis
        self.num_rel = num_rel

        self.comp_fn = comp_fn
        self.dropout = dropout
        self.activation = activation
        self.batchnorm = batchnorm

        # Input layer, complete two tasks
        # 1. define basis layer
        self.basis = dglnn.WeightBasis([self.rel_emb_dim], self.num_basis, self.num_rel)

        # 2. define matrices for convert node dimensions to relation embedding dimensions
        self.input_layer = {}
        for ntype, in_feat in in_feat_dict.items():
            self.input_layer[ntype] = nn.Parameter(th.Tensor(in_feat, self.rel_emb_dim))

        # Hidden layers with n - 1 CompGraphConv layers
        self.layers = nn.ModuleList()
        for l in range(1, (self.num_layer - 1)):
            self.layers.append(CompGraphConv(self.rel_emb_dim,
                                             comp_fn=self.comp_fn,
                                             activation=self.activation,
                                             bias=True,
                                             dropout=self.dropout,
                                             batchnorm=self.batchnorm))

        # Output layer with the output class
        self.output_layer = nn.Parameter(th.Tensor(self.rel_emb_dim, self.out_feat))

        # initialize input and output weights
        for ntype, input_layer in self.input_layer.items():
            nn.init.xavier_uniform_(input_layer)

        nn.init.xavier_uniform_(self.output_layer)

    def forward(self, graph, nfeats):

        # Compute relation input
        h_e = self.basis.forward()

        # Convert node input dimension to relation dimension
        h_n = {}
        for ntype, feat in nfeats.items():
            h_n[ntype] = th.matmul(feat, self.input_layer[ntype])

        # Forward of n layers of CompGraphConv
        for layer in self.layers:
            h_n, h_e = layer(graph, h_n, h_e)

        # Forward of output layer
        outputs = {}
        for ntype, h in h_n.items():
            outputs[ntype] = th.matmul(h, self.output_layer)

        return outputs


def build_dummy_comp_data():
    """
    This dummy data simulate the graph in CompGCN paper figure 1. Here there are 4 types of nodes:
    1. User, e.g. Christopher Nolan
    2. City, e.g. City of London
    3. Country, e.g. United Kindom
    4. Film, e.g. Dark Knight
    The figure 1 is very simple, one node of each type, and only contains 3 relations among them
    1. Born_in, e.g. Nolan was born_in City of London
    2. Citizen_of, e.g. Nolan is citizen_of United Kingdom
    3. Directed_by, e.g. Film Dark Knight is directed_by Nolan

    Returns
    -------
    A DGLGraph with 4 nodes of 4 types, and 3 edges in 3 types.

    """
    g = dgl.heterograph(
        {
         ('user', 'born_in', 'city'): ([th.tensor(0)], [th.tensor(0)]),
         ('user', 'citizen_of', 'country'): ([th.tensor(0)], [th.tensor(0)]),
         ('film', 'directed_by', 'user'): ([th.tensor(0), th.tensor(1)], [th.tensor(0),th.tensor(0)]),
         # add inversed edges
         ('city', 'born_in_inv', 'user'): ([th.tensor(0)], [th.tensor(0)]),
         ('country', 'citizen_of_inv', 'user'): ([th.tensor(0)], [th.tensor(0)]),
         ('user', 'directed_by_inv', 'film'): ([th.tensor(0), th.tensor(0)], [th.tensor(0), th.tensor(1)])
        }
    )

    n_feats = {
        'user': th.ones(1, 5),
        'city': th.ones(1, 5) * 2,
        'country': th.ones(1, 5) * 4,
        'film': th.ones(2, 5) * 8
    }

    e_feats = {
        'born_in': th.ones(1, 5) * 0.5,
        'citizen_of': th.ones(1, 5) * 0.5 * 0.5,
        'directed_by': th.ones(2, 5) * 0.5 * 0.5 * 0.5,
        'born_in_inv': th.ones(1, 5) * 0.5,
        'citizen_of_inv': th.ones(1, 5) * 0.5 * 0.5,
        'directed_by_inv': th.ones(2, 5) * 0.5 * 0.5 * 0.5
    }

    return g, n_feats, e_feats


if __name__ == '__main__':
    graph, n_feats, e_feats = build_dummy_comp_data()

    # Test for one layer of CompGraphConv module
    # compgcn = CompGraphConv(hid_dim=5,
    #                         comp_fn='mul',
    #                         activation=nn.functional.relu,
    #                         bias=True,
    #                         dropout=0.1,
    #                         batchnorm=False)
    # nfeats, efeats = compgcn(graph, n_feats, e_feats)
    # for k, v in nfeats.items():
    #     print(k)
    #     print(v)
    # for k, v in efeats.items():
    #     print(k)
    #     print(v)

    # Test for one CompGCN module
    in_feat_dict = {
        'user': 5,
        'city': 5,
        'country': 5,
        'film': 5
    }
    compgcn_model = CompGCN(in_feat_dict=in_feat_dict,
                            hid_dim=3,
                            num_layers=2,
                            out_feat=2,
                            num_basis=2,
                            num_rel=6,
                            comp_fn='sub',
                            dropout=0.0,
                            activation=None,
                            batchnorm=False
                            )
    logits = compgcn_model.forward(graph, n_feats)
    print(logits)