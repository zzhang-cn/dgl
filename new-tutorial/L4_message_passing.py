"""
Writing GNN Modules for Stochastic GNN Training
===============================================

All GNN modules DGL provides support stochastic GNN training. This
tutorial teaches you how to write your own graph neural network module
for stochastic GNN training. It assumes that

1. You know :doc:`how to write GNN modules for full graph
   training <3_message_passing>`.
2. You know :doc:`how stochastic GNN training pipeline
   works <L1_large_node_classification>`.

"""

import dgl
import torch
import numpy as np
from ogb.nodeproppred import DglNodePropPredDataset

dataset = DglNodePropPredDataset('ogbn-products')

graph, node_labels = dataset[0]
idx_split = dataset.get_idx_split()
train_nids = idx_split['train']
node_features = graph.ndata['feat']

sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
train_dataloader = dgl.dataloading.NodeDataLoader(
    graph, train_nids, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0
)

input_nodes, output_nodes, bipartites = next(iter(train_dataloader))


######################################################################
# DGL Bipartite Graph Introduction
# --------------------------------
#
# In the previous tutorials, you have seen the concept *bipartite graph*,
# where nodes are divided into two parts.
# This section introduces how you can manipulate (directional) bipartite
# graphs.
#
# You can access the input node features and output node features via
# ``srcdata`` and ``dstdata`` attributes:
#

bipartite = bipartites[0]
print(bipartite.srcdata)
print(bipartite.dstdata)


######################################################################
# It also has ``num_src_nodes`` and ``num_dst_nodes`` functions to query
# how many input nodes and output nodes exist in the bipartite graph:
#

print(bipartite.num_src_nodes(), bipartite.num_dst_nodes())


######################################################################
# You can assign features to ``srcdata`` and ``dstdata`` just as what you
# will do with ``ndata`` on the graphs you have seen earlier:
#

bipartite.srcdata['x'] = torch.zeros(bipartite.num_src_nodes(), bipartite.num_dst_nodes())
dst_feat = bipartite.dstdata['feat']


######################################################################
# Also, since the bipartite graphs are constructed by DGL, you can
# retrieve the input node IDs (i.e. those that are required to compute the
# output) and output node IDs (i.e. those whose representations the
# current GNN layer should compute) as follows.
#

bipartite.srcdata[dgl.NID], bipartite.dstdata[dgl.NID]


######################################################################
# Writing GNN Modules for Bipartite Graphs for Stochastic Training
# ----------------------------------------------------------------
#


######################################################################
# Recall that the bipartite graphs yielded by the ``NodeDataLoader`` and
# ``EdgeDataLoader`` have the property that the first few input nodes are
# always identical to the output nodes:
#
# |image1|
#
# .. |image1| image:: https://data.dgl.ai/tutorial/img/bipartite.gif
#

print(torch.equal(bipartite.srcdata[dgl.NID][:bipartite.num_dst_nodes()], bipartite.dstdata[dgl.NID]))


######################################################################
# Suppose you have obtained the input node representations
# :math:`h_u^{(l-1)}`:
#

bipartite.srcdata['h'] = torch.randn(bipartite.num_src_nodes(), 10)


######################################################################
# Recall that DGL provides the `update_all` interface for expressing how
# to compute messages and how to aggregate them on the nodes that receive
# them. This concept naturally applies to bipartite graphs -- message
# computation happens on the edges between source and destination nodes of
# the edges, and message aggregation happens on the destination nodes.
#
# For example, suppose the message function copies the source feature
# (i.e. :math:`M^{(l)}\left(h_v^{(l-1)}, h_u^{(l-1)}, e_{u\to v}^{(l-1)}\right) = h_v^{(l-1)}`),
# and the reduce function averages the received messages.  Performing
# such message passing computation on a bipartite graph is no different than
# on a full graph:
#

import dgl.function as fn

bipartite.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h'))
m_v = bipartite.dstdata['h']
m_v


######################################################################
# Putting them together, you can implement a GraphSAGE convolution for
# training with neighbor sampling as follows (the differences to the :doc:`full graph
# counterpart <3_message_passing>` are highlighted with arrows ``<---``)
#

import torch.nn as nn
import torch.nn.functional as F
import tqdm

class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input bipartite graph.
        h : (Tensor, Tensor)
            The feature of input nodes and output nodes as a pair of Tensors.
        """
        with g.local_scope():
            h_src, h_dst = h
            g.srcdata['h'] = h_src                        # <---
            g.dstdata['h'] = h_dst                        # <---
            # update_all is a message passing API.
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_N = g.dstdata['h_N']
            h_total = torch.cat([h_dst, h_N], dim=1)      # <---
            return self.linear(h_total)

class Model(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes):
        super(Model, self).__init__()
        self.conv1 = SAGEConv(in_feats, h_feats)
        self.conv2 = SAGEConv(h_feats, num_classes)

    def forward(self, bipartites, x):
        h_dst = x[:bipartites[0].num_dst_nodes()]
        h = self.conv1(bipartites[0], (x, h_dst))
        h = F.relu(h)
        h_dst = h[:bipartites[1].num_dst_nodes()]
        h = self.conv2(bipartites[1], (h, h_dst))
        return h

sampler = dgl.dataloading.MultiLayerNeighborSampler([4, 4])
train_dataloader = dgl.dataloading.NodeDataLoader(
    graph, train_nids, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
    num_workers=0
)
model = Model(graph.ndata['feat'].shape[1], 128, dataset.num_classes).cuda()

with tqdm.tqdm(train_dataloader) as tq:
    for step, (input_nodes, output_nodes, bipartites) in enumerate(tq):
        bipartites = [b.to(torch.device('cuda')) for b in bipartites]
        inputs = node_features[input_nodes].cuda()
        labels = node_labels[output_nodes].cuda()
        predictions = model(bipartites, inputs)


######################################################################
# Both ``update_all`` and the functions in ``nn.functional`` namespace
# support bipartite graphs, so you can migrate the code working for small
# graphs to large graph training with minimal changes introduced above.
#


######################################################################
# Writing GNN Modules for Both Full-graph Training and Stochastic Training
# ------------------------------------------------------------------------
#
# Here is a step-by-step tutorial for writing a GNN module for both
# :doc:`full-graph training <1_introduction>` *and* :doc:`stochastic
# training <L1_node_classification>`.
#
# Say you start with a GNN module that works for full-graph training only:
#

class SAGEConv(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super().__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor
            The input node feature.
        """
        with g.local_scope():
            g.ndata['h'] = h
            # update_all is a message passing API.
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h, h_N], dim=1)
            return self.linear(h_total)


######################################################################
# **First step**: Check whether the input feature is a single tensor or a
# pair of tensors:
#
# .. code:: python
#
#    if isinstance(h, tuple):
#        h_src, h_dst = h
#    else:
#        h_src = h_dst = h
#
# **Second step**: Replace node features ``h`` with ``h_src`` or
# ``h_dst``, and assign the node features to ``srcdata`` or ``dstdata``,
# instead of ``ndata``.
#
# Whether to assign to ``srcdata`` or ``dstdata`` depends on whether the
# said feature acts as the features on source nodes or destination nodes
# of the edges in the message functions (in ``update_all`` or
# ``apply_edges``).
#
# *Example 1*: For the following ``update_all`` statement:
#
# .. code:: python
#
#    g.ndata['h'] = h
#    g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
#
# The node feature ``h`` acts as source node feature because ``'h'``
# appeared as source node feature. So you will need to replace ``h`` with
# source feature ``h_src`` and assign to ``srcdata`` for the version that
# works with both cases:
#
# .. code:: python
#
#    g.srcdata['h'] = h_src
#    g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
#
# *Example 2*: For the following ``apply_edges`` statement:
#
# .. code:: python
#
#    g.ndata['h'] = h
#    g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
#
# The node feature ``h`` acts as both source node feature and destination
# node feature. So you will assign ``h_src`` to ``srcdata`` and ``h_dst``
# to ``dstdata``:
#
# .. code:: python
#
#    g.srcdata['h'] = h_src
#    g.dstdata['h'] = h_dst
#    # The first 'h' corresponds to source feature (u) while the second 'h' corresponds to destination feature (v).
#    g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
#
# .. note::
#
#    For homogeneous graphs (i.e. graphs with only one node type
#    and one edge type), ``srcdata`` and ``dstdata`` are aliases of
#    ``ndata``. So you can safely replace ``ndata`` with ``srcdata`` and
#    ``dstdata`` even for full-graph training.
#
# **Third step**: Replace the ``ndata`` for outputs with ``dstdata``.
#
# For example, the following code
#
# .. code:: python
#
#    # Assume that update_all() function has been called with output node features in `h_N`.
#    h_N = g.ndata['h_N']
#    h_total = torch.cat([h, h_N], dim=1)
#
# will change to
#
# .. code:: python
#
#    h_N = g.dstdata['h_N']
#    h_total = torch.cat([h_dst, h_N], dim=1)
#


######################################################################
# Putting together, you will change the ``SAGEConvForBoth`` module above
# to something like the following:
#

class SAGEConvForBoth(nn.Module):
    """Graph convolution module used by the GraphSAGE model.

    Parameters
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    """
    def __init__(self, in_feat, out_feat):
        super().__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, g, h):
        """Forward computation

        Parameters
        ----------
        g : Graph
            The input graph.
        h : Tensor or tuple[Tensor, Tensor]
            The input node feature.
        """
        with g.local_scope():
            if isinstance(h, tuple):
                h_src, h_dst = h
            else:
                h_src = h_dst = h

            g.srcdata['h'] = h_src
            # update_all is a message passing API.
            g.update_all(message_func=fn.copy_u('h', 'm'), reduce_func=fn.mean('m', 'h_N'))
            h_N = g.ndata['h_N']
            h_total = torch.cat([h_dst, h_N], dim=1)
            return self.linear(h_total)

