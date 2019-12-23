"""
.. _model-gcn:

Graph convolutional network
====================================

**Author:** `Qi Huang <https://github.com/HQ01>`_, `Minjie Wang  <https://jermainewang.github.io/>`_,
Yu Gai, Quan Gan, Zheng Zhang

This introduction to using Deep Graph Library (DGL) to implement graph convolutional neural 
networks (GCN), builds upon the :doc:`PageRank with DGL Message Passing <../../basics/3_pagerank>` tutorial. 
Here you learn how DGL combines graph with a deep neural network and learn structural representations.

To learn more about the research that preceeds DGL, see `Semi-Supervised Classification with Graph
Convolutional Networks <https://arxiv.org/pdf/1609.02907.pdf>`_.
"""

###############################################################################
# GCN from the perspective of message passing
# ```````````````````````````````````````````````
# You can approach a GCN from a message
# passing perspective. It can be summarized as the following steps: For each node :math:`u`:
# 
# First, aggregate neighbors' representations :math:`h_{v}` to produce an
# intermediate representation :math:`\hat{h}_u`.  
# Second, Transform the aggregated
# representation :math:`\hat{h}_{u}` with a linear projection followed by a
# non-linearity: :math:`h_{u} = f(W_{u} \hat{h}_u)`.
# 
# Implement step 1 with DGL message passing, and step 2 with the
# ``apply_nodes`` method, whose node UDF is a PyTorch ``nn.Module``.
# 
# To learn more about the `mathematical formula <math_>`_ skip ahead to the end of this section.
#
# GCN implementation with DGL
# ``````````````````````````````````````````
# First define the message and reduce the function.  Since the
# aggregation on a node :math:`u` only involves summing over the neighbors'
# representations :math:`h_v`, we can simply use built-in functions.

import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph

gcn_msg = fn.copy_src(src='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')

###############################################################################
# Define the node UDF for ``apply_nodes``, which is a fully-connected layer.

class NodeApplyModule(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(NodeApplyModule, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)
        self.activation = activation

    def forward(self, node):
        h = self.linear(node.data['h'])
        if self.activation is not None:
            h = self.activation(h)
        return {'h' : h}

###############################################################################
# Define the GCN module. A GCN layer essentially performs
# message passing on all the nodes then applies the `NodeApplyModule`. 
# The dropout in the paper is omitted for simplicity.

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats, activation):
        super(GCN, self).__init__()
        self.apply_mod = NodeApplyModule(in_feats, out_feats, activation)

    def forward(self, g, feature):
        g.ndata['h'] = feature
        g.update_all(gcn_msg, gcn_reduce)
        g.apply_nodes(func=self.apply_mod)
        return g.ndata.pop('h')

###############################################################################
# The forward function is essentially the same as any other commonly seen NNs
# model in PyTorch. You can initialize GCN like any ``nn.Module``. For example,
# define a simple neural network consisting of two GCN layers. Suppose you
# are training the classifier for the Cora dataset. The input feature size is
# 1433 and the number of classes is 7. The last GCN layer computes node embeddings,
# so the last layer in general doesn't apply activation.

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gcn1 = GCN(1433, 16, F.relu)
        self.gcn2 = GCN(16, 7, None)
    
    def forward(self, g, features):
        x = self.gcn1(g, features)
        x = self.gcn2(g, x)
        return x
net = Net()
print(net)

###############################################################################
# Load the cora dataset using DGL's built-in data module.

from dgl.data import citation_graph as citegrh
import networkx as nx
def load_cora_data():
    data = citegrh.load_cora()
    features = th.FloatTensor(data.features)
    labels = th.LongTensor(data.labels)
    train_mask = th.ByteTensor(data.train_mask)
    test_mask = th.ByteTensor(data.test_mask)
    g = data.graph
    # add self loop
    g.remove_edges_from(nx.selfloop_edges(g))
    g = DGLGraph(g)
    g.add_edges(g.nodes(), g.nodes())
    return g, features, labels, train_mask, test_mask

###############################################################################
# When a model is trained, we can use the following method to evaluate
# the performance of the model on the test dataset:

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

###############################################################################
# Train the network as follows:

import time
import numpy as np
g, features, labels, train_mask, test_mask = load_cora_data()
optimizer = th.optim.Adam(net.parameters(), lr=1e-3)
dur = []
for epoch in range(50):
    if epoch >=3:
        t0 = time.time()

    net.train()
    logits = net(g, features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[train_mask], labels[train_mask])
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if epoch >=3:
        dur.append(time.time() - t0)
    
    acc = evaluate(net, g, features, labels, test_mask)
    print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))

###############################################################################
# .. _math:
#
# GCN in one formula
# ```````````````````````````````````````````````
# Mathematically, the GCN model follows this formula.
# 
# :math:`H^{(l+1)} = \sigma(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}H^{(l)}W^{(l)})`
# 
# Here, :math:`H^{(l)}` denotes the :math:`l^{th}` layer in the network,
# :math:`\sigma` is the non-linearity, and :math:`W` is the weight matrix for
# this layer. :math:`D` and :math:`A`, as commonly seen, represent degree
# matrix and adjacency matrix, respectively. The ~ is a renormalization trick
# in which you add a self-connection to each node of the graph, and build the
# corresponding degree and adjacency matrix. The shape of the input
# :math:`H^{(0)}` is :math:`N \times D`, where :math:`N` is the number of nodes
# and :math:`D` is the number of input features. Chain-up multiple
# layers as such to produce a node-level representation output with shape
# :math`N \times F`, where :math:`F` is the dimension of the output node
# feature vector.
# 
# The equation can be efficiently implemented using sparse matrix
# multiplication kernels, such as Kipf's
# `pygcn <https://github.com/tkipf/pygcn>`_ code). The above DGL implementation
# has already used this due to the use of built-in functions. To
# learn more about the background, see the tutorial on :doc:`PageRank <../../basics/3_pagerank>`.
