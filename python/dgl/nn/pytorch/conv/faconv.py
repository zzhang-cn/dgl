"""Torch Module for Frequency Adaptive Graph Convolutional layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch as th
from torch import nn
from torch.nn import functional as F

from .... import function as fn


class FAConv(nn.Module):
    r"""

    Description
    -----------
    The Frequency Adaptation Graph Convolutional Networks (FAGCN) was introduced in
    `"Beyond Low-frequency Information in Graph Convolutional Networks"
    <https://arxiv.org/pdf/2101.00797.pdf>`__. paper
    and mathematically is defined as follows:
    .. math::
        \mathbf{h}^{(0)}_i= \phi(\mathbf{W}_1\mathbf{h}_i)
    .. math::
        \mathbf{h}^{(l+1)}_i= \epsilon \cdot \mathbf{h}^{(0)}_i +
        \sum_{j \in \mathcal{N}(i)} \frac{\alpha_{i,j}^G}{\sqrt{d_i d_j}}\mathbf{h}_{j}^{(l)}
    .. math::
        \mathbf{h}_{out}= \mathbf{W}_2\mathbf{h}_j^{(L)}
    .. math::
        \alpha_{i,j}^G = \tanh(\mathbf{g}^T \mathbf{h}_{i}||\mathbf{h}_{j}])

    where :math:`\mathbf{h}_i` denotes initial features,
    :math:`\phi` activation function ,
    :math:`d_i` is node degree of a node,
    :math:`\mathbf{W}_1,\mathbf{W}_2,` is are weight matrices
    :math:` \mathbf{g} \in \mathbb{R}^{2F}` is  a shared convolutional kernel

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    hidden_feats : int
      Size for the hidden representations.
    out_feats : int
        Size for the output representations:math:`h_j^{(l+1)}`.
    eps : float, optional
        :math:`\epsilon`-value. Default: ``0.1``
    num_layers : int
        Number of  layers, ranging from 1 to :math:`L`
    dropout : float, optional
        Dropout rate.Default: ``0.0``
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GCN2Conv

    >>> # Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 3)
    >>> conv = FAConv (in_feats=3,hidden_feats=10, out_feats=3,
    num_layers=2,dropout=0.0,bias=True, activation=None)
    >>> res = conv(g, feat)
    >>> print(res)
    tensor([[ 1.7546,  0.1991,  1.4269],
        [ 1.7546,  0.1991,  1.4269],
        [ 1.7546,  0.1991,  1.4269],
        [ 2.6728,  0.2279,  2.1793],
        [ 1.1053,  0.1787,  0.8948],
        [-0.4622,  0.1294, -0.3897]], grad_fn=<AddBackward0>)

    """

    def __init__(self,
                 in_feats,
                 hidden_feats,
                 out_feats,
                 eps=0.1,
                 num_layers=2,
                 dropout=0.0,
                 bias=True,
                 activation=None):

        super(FAConv, self).__init__()

        self._in_feats = in_feats
        self._hidden_feats = hidden_feats
        self._out_feats = out_feats
        self._eps = eps
        self._dropout = dropout
        self._bias = bias
        self._activation = activation
        self._num_layers = num_layers

        self.gate = nn.Linear(2 * self._hidden_feats, 1)
        self.W1 = nn.Linear(self._in_feats, self._hidden_feats)
        self.W2 = nn.Linear(self._hidden_feats, self._out_feats)

        if bias:
            self.bias = nn.Parameter(th.Tensor(out_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()
        self._activation = activation

    def reset_parameters(self):

        r"""

        Description
        -----------
        Reinitialize learnable parameters.
        """

        nn.init.xavier_normal_(self.gate.weight, gain=1.414)
        nn.init.xavier_normal_(self.W1.weight, gain=1.414)
        nn.init.xavier_normal_(self.W2.weight, gain=1.414)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, graph, feat):
        r"""

        Description
        -----------
        Compute Frequency Adaptation Graph Convolutional layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.

        """
        with graph.local_scope():

            deg = graph.in_degrees().to(graph.device).float().clamp(min=1)
            norm = th.pow(deg, -0.5)
            graph.ndata["d"] = norm

            feat = F.dropout(feat, p=self._dropout, training=self.training)
            h = th.relu(self.W1(feat))
            h = F.dropout(h, p=self._dropout, training=self.training)
            raw = h

            graph.ndata["h"] = h

            for _ in range(self._num_layers):
                graph.apply_edges(self.edge_udf)
                graph.update_all(fn.u_mul_e("h", "e", "_"), fn.sum("_", "z"))
                if self._eps > 0.0:
                    h = self._eps * raw + graph.dstdata["z"]
                else:
                    h = graph.dstdata["z"]

            rst = self.W2(h)

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def edge_udf(self, edges):
        r"""

        Description
        -----------
        edge udf.

        """
        cat_h = th.cat([edges.dst["h"], edges.src["h"]], dim=1)
        g = th.tanh(self.gate(cat_h)).squeeze()
        e = g * edges.dst["d"] * edges.src["d"]
        e = F.dropout(e, p=self._dropout, training=self.training)

        return {"e": e}

    def extra_repr(self):

        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}, out={_out_feats}, hidden={_hidden_feats}"
        summary += ", num_layers={_num_layers}"
        summary += ", dropout={_dropout}"
        summary += ", eps={_eps}"
        if "_activation" in self.__dict__:
            summary += ", activation={_activation}"

        return summary.format(**self.__dict__)
