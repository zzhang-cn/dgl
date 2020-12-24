"""Torch Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import jax
from jax import numpy as jnp
from flax import linen as nn

from .... import function as fn
from .. import utils


class RelGraphConv(nn.Module):
    r"""

    Description
    -----------
    Relational graph convolution layer.

    Relational graph convolution is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    and can be described as below:

    .. math::

       h_i^{(l+1)} = \sigma(\sum_{r\in\mathcal{R}}
       \sum_{j\in\mathcal{N}^r(i)}\frac{1}{c_{i,r}}W_r^{(l)}h_j^{(l)}+W_0^{(l)}h_i^{(l)})

    where :math:`\mathcal{N}^r(i)` is the neighbor set of node :math:`i` w.r.t. relation
    :math:`r`. :math:`c_{i,r}` is the normalizer equal
    to :math:`|\mathcal{N}^r(i)|`. :math:`\sigma` is an activation function. :math:`W_0`
    is the self-loop weight.

    The basis regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \sum_{b=1}^B a_{rb}^{(l)}V_b^{(l)}

    where :math:`B` is the number of bases, :math:`V_b^{(l)}` are linearly combined
    with coefficients :math:`a_{rb}^{(l)}`.

    The block-diagonal-decomposition regularization decomposes :math:`W_r` into :math:`B`
    number of block diagonal matrices. We refer :math:`B` as the number of bases.

    The block regularization decomposes :math:`W_r` by:

    .. math::

       W_r^{(l)} = \oplus_{b=1}^B Q_{rb}^{(l)}

    where :math:`B` is the number of bases, :math:`Q_{rb}^{(l)}` are block
    bases with shape :math:`R^{(d^{(l+1)}/B)*(d^{l}/B)}`.

    Parameters
    ----------
    in_feat : int
        Input feature size; i.e, the number of dimensions of :math:`h_j^{(l)}`.
    out_feat : int
        Output feature size; i.e., the number of dimensions of :math:`h_i^{(l+1)}`.
    num_rels : int
        Number of relations. .
    regularizer : str
        Which weight regularizer to use "basis" or "bdd".
        "basis" is short for basis-diagonal-decomposition.
        "bdd" is short for block-diagonal-decomposition.
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: ``None``.
    bias : bool, optional
        True if bias is added. Default: ``True``.
    activation : callable, optional
        Activation function. Default: ``None``.
    self_loop : bool, optional
        True to include self loop message. Default: ``True``.
    low_mem : bool, optional
        True to use low memory implementation of relation message passing function. Default: False.
        This option trades speed with memory consumption, and will slowdown the forward/backward.
        Turn it on when you encounter OOM problem during training or evaluation. Default: ``False``.
    dropout : float, optional
        Dropout rate. Default: ``0.0``
    layer_norm: float, optional
        Add layer norm. Default: ``False``

    Examples
    --------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import RelGraphConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = jnp.ones(6, 10)
    >>> conv = RelGraphConv(10, 2, 3, regularizer='basis', num_bases=2)
    >>> conv.weight.shape
    torch.Size([2, 10, 2])
    >>> etype = jnp.tensor(np.array([0,1,2,0,1,2]).astype(np.int64))
    >>> res = conv(g, feat, etype)
    >>> res
    tensor([[ 0.3996, -2.3303],
            [-0.4323, -0.1440],
            [ 0.3996, -2.3303],
            [ 2.1046, -2.8654],
            [-0.4323, -0.1440],
            [-0.1309, -1.0000]], grad_fn=<AddBackward0>)

    >>> # One-hot input
    >>> one_hot_feat = jnp.tensor(np.array([0,1,2,3,4,5]).astype(np.int64))
    >>> res = conv(g, one_hot_feat, etype)
    >>> res
    tensor([[ 0.5925,  0.0985],
            [-0.3953,  0.8408],
            [-0.9819,  0.5284],
            [-1.0085, -0.1721],
            [ 0.5962,  1.2002],
            [ 0.0365, -0.3532]], grad_fn=<AddBackward0>)
    """

    in_feat: int
    out_feat: int
    num_rels: int
    regularizer: str = "basis"
    from typing import Union
    num_bases: Union[int, None] = None
    bias: bool = True
    activation: Union[callable, None] = None
    self_loop: bool = True
    low_mem: bool = False
    dropout: float = 0.0
    layer_norm: bool = False


    def setup(self):
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases <= 0:
            self.num_bases = self.num_rels

        if self.regularizer == "basis":
            # add basis weights
            self.weight = self.param(
                "weight",
                nn.initializers.xavier_uniform(),
                (self.num_bases, self.in_feat, self.out_feat)
            )
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = self.param(
                    "w_comp",
                    nn.initializers.xavier_uniform(),
                    (self.num_rels, self.num_bases)
                )

            # message func
            self.message_func = self.basis_message_func

        elif self.regularizer == "bdd":
            if self.in_feat % self.num_bases != 0 or self.out_feat % self.num_bases != 0:
                raise ValueError(
                    'Feature size must be a multiplier of num_bases (%d).'
                    % self.num_bases
                )
            # add block diagonal weights
            self.submat_in = self.in_feat // self.num_bases
            self.submat_out = self.out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = self.param(
                "weight",
                nn.initializers.xavier_uniform(),
                (self.num_rels, self.num_bases * self.submat_in * self.submat_out),
            )

            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = self.param(
                "h_bias",
                nn.initializers.zeros,
                (self.out_feat,),
            )

        # layer norm
        if self.layer_norm:
            self.layer_norm_weight = nn.LayerNorm()

        # weight for self loop
        if self.self_loop:
            self.loop_weight = self.param(
                "loop_weight",
                nn.initializers.xavier_uniform(),
                (self.in_feat, self.out_feat),
            )

        if self.dropout > 0.0:
            self._dropout = nn.Dropout(self.dropout)
        else:
            self._dropout = lambda x: x

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.reshape((self.num_bases,
                                      self.in_feat * self.out_feat))
            weight = jnp.matmul(self.w_comp, weight).reshape((
                self.num_rels, self.in_feat, self.out_feat))
        else:
            weight = self.weight

        # calculate msg @ W_r before put msg into edge
        # if src is jnp.int64 we expect it is an index select
        if edges.src['h'].dtype != jnp.int64 and self.low_mem:
            etypes = jnp.unique(edges.data['type'])
            msg = jnp.zeros((edges.src['h'].shape[0], self.out_feat))
            for etype in etypes:
                loc = edges.data['type'] == etype
                w = weight[etype]
                src = edges.src['h'][loc]
                sub_msg = jnp.matmul(src, w)
                msg = jax.ops.index_update(
                    msg,
                    loc,
                    sub_msg
                )
        else:
            # put W_r into edges then do msg @ W_r
            msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'])

        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        if edges.src['h'].dtype == jnp.int64 and len(edges.src['h'].shape) == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')

        # calculate msg @ W_r before put msg into edge
        if self.low_mem:
            etypes = jnp.unique(edges.data['type'])
            msg = jnp.zeros((edges.src['h'].shape[0], self.out_feat))
            for etype in etypes:
                loc = edges.data['type'] == etype
                w = self.weight[etype].reshape((self.num_bases, self.submat_in, self.submat_out))
                src = edges.src['h'][loc].reshape((-1, self.num_bases, self.submat_in))
                sub_msg = jnp.einsum('abc,bcd->abd', src, w)
                sub_msg = sub_msg.reshape((-1, self.out_feat))
                msg = jax.ops.index_update(
                    msg,
                    loc,
                    sub_msg
                )
        else:
            weight = jnp.take(
                self.weight,
                edges.data['type'],
                0,
            ).reshape(
                (-1, self.submat_in, self.submat_out),
            )

            node = edges.src['h'].reshape((-1, 1, self.submat_in))
            msg = jax.lax.batch_matmul(node, weight).reshape((-1, self.out_feat))
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def __call__(self, g, feat, etypes, norm=None):
        """

        Description
        -----------

        Forward computation

        Parameters
        ----------
        g : DGLGraph
            The graph.
        feat : torch.Tensor
            Input node features. Could be either

                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. It then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`.

        Returns
        -------
        torch.Tensor
            New node features.
        """
        with g.local_scope():
            g.srcdata['h'] = feat
            g.edata['type'] = etypes
            if norm is not None:
                g.edata['norm'] = norm
            if self.self_loop:
                loop_message = utils.matmul_maybe_select(feat[:g.number_of_dst_nodes()],
                                                         self.loop_weight)

            # message passing
            g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
            # apply bias and activation
            node_repr = g.dstdata['h']
            if self.layer_norm:
                node_repr = self.layer_norm_weight(node_repr)
            if self.bias:
                node_repr = node_repr + self.h_bias
            if self.self_loop:
                node_repr = node_repr + loop_message
            if self.activation:
                node_repr = self.activation(node_repr)
            node_repr = self._dropout(node_repr)
            return node_repr
