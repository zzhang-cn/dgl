"""Torch Module for Topology Adaptive Graph Convolutional layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import math

import torch as th
from torch import nn

from .... import function as fn
from ....base import DGLError


class GCN2Conv(nn.Module):

    r"""

    Description
    -----------
    The Graph Convolutional Network via Initial residual
    and Identity mapping (GCNII) was introduced in `"Simple and Deep Graph Convolutional
    Networks" <https://arxiv.org/abs/2007.02133>`_ paper
    and mathematically is defined as follows:
    .. math::
        \mathbf{h}^{(l+1)} =\left( (1 - \alpha)(\mathbf{D}^{-1/2} \mathbf{\hat{A}}
        \mathbf{D}^{-1/2})\mathbf{h}^{(l)} + \alpha {\mathbf{h}^{(0)}} \right)
        \left( (1 - \beta) \mathbf{I} + \beta \mathbf{W} \right)

    where :math:`\mathbf{\hat{A}}` denotes the adjacency matrix with self-loops,
    :math:`\mathbf{D}_{ii} = \sum_{j=0} \mathbf{A}_{ij}` its diagonal degree matrix ,
    :math:`\mathbf{h}^{(0)}` is initial features,
    :math:`\mathbf{h}^{(l)}` is feature of the current layer
    :math:` \alpha` for  fraction of initial features
    :math:`\hyper-lambda` hyperparameter to tune strength of indentity mapping

    Parameters
    ----------
    in_feats : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    alpha : float
        a fraction of the initial input feature. Default: ``0.1``
    hyper_lambda : float
        a hypermeter to ensure the decay of the weight matrix
         adaptively increases. Default: ``1``
    layer : int
        the index of current layer. Default: ``1``
    share_weight : bool
        Whether to share a weight matrix between initial features and
        smoothed features. Default: ``False``
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: ``True``.
    activation : callable activation function/layer or None, optional
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    allow_zero_in_degree : bool, optional
        If there are 0-in-degree nodes in the graph, output for those nodes will be invalid
        since no message will be passed to those nodes. This is harmful for some applications
        causing silent performance regression. This module will raise a DGLError if it detects
        0-in-degree nodes in input graph. By setting ``True``, it will suppress the check
        and let the users handle it by themselves. Default: ``False``.

    Note
    ----
    Zero in-degree nodes will lead to invalid output value. This is because no message
    will be passed to those nodes, the aggregation function will be appied on empty input.
    A common practice to avoid this is to add a self-loop for each node in the graph if
    it is homogeneous, which can be achieved by:

    >>> g = ... # a DGLGraph
    >>> g = dgl.add_self_loop(g)

    Calling ``add_self_loop`` will not work for some graphs, for example, heterogeneous graph
    since the edge type can not be decided for self_loop edges. Set ``allow_zero_in_degree``
    to ``True`` for those cases to unblock the code and handle zero-in-degree nodes manually.
    A common practise to handle this is to filter out the nodes with zero-in-degree when use
    after conv.

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GCN2Conv

    >>> # Homogeneous graph
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 3)
    >>> g=dgl.add_self_loop(g)
    >>> conv = GCN2Conv(3,alpha=0.5,share_weights=True,allow_zero_in_degree=True)
    >>> res = conv(g, feat, feat/2.)
    >>> print(res)
    tensor([[2.6484, 0.8584, 0.9750],
            [2.6484, 0.8584, 0.9750],
            [2.6484, 0.8584, 0.9750],
            [2.9989, 0.8283, 0.9696],
            [2.5476, 0.8670, 0.9765],
            [2.6484, 0.8584, 0.9750]], grad_fn=<AddBackward0>)

    """

    def __init__(self,
                 in_feats,
                 alpha=0.1,
                 hyper_lambda=1,
                 layer=1,
                 share_weights=True,
                 allow_zero_in_degree=False,
                 bias=True,
                 activation=None):
        super().__init__()

        self._in_feats = in_feats
        self._share_weights = share_weights

        self.alpha = alpha
        self.beta = math.log(hyper_lambda / layer + 1)

        self._bias = bias
        self._activation = activation
        self._allow_zero_in_degree = allow_zero_in_degree

        self.weight1 = nn.Parameter(th.Tensor(self._in_feats, self._in_feats))

        if self._share_weights:
            self.register_parameter("weight2", None)
        else:
            self.weight2 = nn.Parameter(th.Tensor(self._in_feats, self._in_feats))

        if self._bias:
            self.bias = nn.Parameter(th.Tensor(self._in_feats))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        r"""

        Description
        -----------
        Reinitialize learnable parameters.

        """
        nn.init.normal_(self.weight1)
        if not self._share_weights:
            nn.init.normal_(self.weight2)
        if self._bias is not None:
            nn.init.zeros_(self.bias)

    def set_allow_zero_in_degree(self, set_value):
        r"""

        Description
        -----------
        Set allow_zero_in_degree flag.

        Parameters
        ----------
        set_value : bool
            The value to be set to the flag.
        """
        self._allow_zero_in_degree = set_value

    def forward(self, graph, feat, feat_0):
        r"""

        Description
        -----------
        Compute graph convolution.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
             it represents the input feature of shape
            :math:`(N, D_{in})`
            where :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
        feat_0 : torch.Tensor
                it represents the initial feature of shape :math:`(N, D_{in})`

        Returns
        -------
        torch.Tensor
            The output feature

        Raises
        ------
        DGLError
            Case 1:
            If there are 0-in-degree nodes in the input graph, it will raise DGLError
            since no message will be passed to those nodes. This will cause invalid output.
            The error can be ignored by setting ``allow_zero_in_degree`` parameter to ``True``.

        Note
        ----
        * Input shape: :math:`(N, *, \text{in_feats})` where * means any number of additional
          dimensions, :math:`N` is the number of nodes.
        * Output shape: :math:`(N, *, \text{out_feats})` where all but the last dimension are
          the same shape as the input.
        * Weight shape: :math:`(\text{in_feats}, \text{out_feats})`.
        """
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError(
                        "There are 0-in-degree nodes in the graph, "
                        "output for those nodes will be invalid. "
                        "This is harmful for some applications, "
                        "causing silent performance regression. "
                        "Adding self-loop on the input graph by "
                        "calling `g = dgl.add_self_loop(g)` will resolve "
                        "the issue. Setting ``allow_zero_in_degree`` "
                        "to be `True` when constructing this module will "
                        "suppress the check and let the code run."
                    )

            # normalize  to get smoothed representation
            degs = graph.in_degrees().float().clamp(min=1)
            norm = th.pow(degs, -0.5)
            norm = norm.to(feat.device).unsqueeze(1)

            feat = feat * norm
            graph.ndata["h"] = feat
            graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
            feat = graph.ndata.pop("h")
            feat = feat * norm
            # scale
            feat = feat * (1 - self.alpha)

            # initial residual connection to the first layer
            feat_0 = feat_0[: feat.size(0)] * self.alpha

            if self._share_weights:
                rst = feat.add_(feat_0)
                rst = th.addmm(
                    feat, feat, self.weight1, beta=(1 - self.beta), alpha=self.beta
                )
            else:
                rst = th.addmm(
                    feat, feat, self.weight1, beta=(1 - self.beta), alpha=self.beta
                )
                rst += th.addmm(
                    feat_0, feat_0, self.weight2, beta=(1 - self.beta), alpha=self.beta
                )

            if self._bias is not None:
                rst = rst + self._bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst

    def extra_repr(self):
        """Set the extra representation of the module,
        which will come into effect when printing the model.
        """
        summary = "in={_in_feats}"
        summary += ", share_weight={_share_weights}, alpha={alpha}, beta={beta}"
        if "self._bias" in self.__dict__:
            summary += ", bias={bias}"
        if "self._activation" in self.__dict__:
            summary += ", activation={_activation}"

        return summary.format(**self.__dict__)
