"""Torch modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ, invalid-name, W0235
import jax
from jax import numpy as jnp
from flax import linen as nn
import numpy as np

from ...backend import jax as F
from ...readout import sum_nodes, mean_nodes, max_nodes, broadcast_nodes,\
    softmax_nodes, topk_nodes


__all__ = ['SumPooling', 'AvgPooling', 'MaxPooling', 'SortPooling',
           'GlobalAttentionPooling', 'Set2Set',
           'SetTransformerEncoder', 'SetTransformerDecoder', 'WeightAndSum']

class SumPooling(object):
    r"""

    Description
    -----------
    Apply sum pooling over the nodes in a graph .

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i} x^{(i)}_k

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn.pytorch.glob import SumPooling
    >>>
    >>> g1 = dgl.DGLGraph()
    >>> g1.add_nodes(2)
    >>> g1_node_feats = jnp.ones(2,5)
    >>>
    >>> g2 = dgl.DGLGraph()
    >>> g2.add_nodes(3)
    >>> g2_node_feats = jnp.ones(3,5)
    >>>
    >>> sumpool = SumPooling()

    Case 1: Input a single graph

    >>> sumpool(g1, g1_node_feats)
        tensor([[2., 2., 2., 2., 2.]])

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = jnp.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> sumpool(batch_g, batch_f)
        tensor([[2., 2., 2., 2., 2.],
                [3., 3., 3., 3., 3.]])
    """

    def __call__(self, graph, feat):
        r"""

        Compute sum pooling.

        Parameters
        ----------
        graph : DGLGraph
            a DGLGraph or a batch of DGLGraphs
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the number
            of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to the
            batch size of input graphs.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = sum_nodes(graph, 'h')
            return readout


class AvgPooling(object):
    r"""

    Description
    -----------
    Apply average pooling over the nodes in a graph.

    .. math::
        r^{(i)} = \frac{1}{N_i}\sum_{k=1}^{N_i} x^{(i)}_k

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn.pytorch.glob import AvgPooling
    >>>
    >>> g1 = dgl.DGLGraph()
    >>> g1.add_nodes(2)
    >>> g1_node_feats = jnp.ones(2,5)
    >>>
    >>> g2 = dgl.DGLGraph()
    >>> g2.add_nodes(3)
    >>> g2_node_feats = jnp.ones(3,5)
    >>>
    >>> avgpool = AvgPooling()

    Case 1: Input single graph

    >>> avgpool(g1, g1_node_feats)
        tensor([[1., 1., 1., 1., 1.]])

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' note features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = jnp.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> avgpool(batch_g, batch_f)
        tensor([[1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.]])
    """

    def __call__(self, graph, feat):
        r"""

        Compute average pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the number
            of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where
            :math:`B` refers to the batch size of input graphs.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = mean_nodes(graph, 'h')
            return readout


class MaxPooling(object):
    r"""

    Description
    -----------
    Apply max pooling over the nodes in a graph.

    .. math::
        r^{(i)} = \max_{k=1}^{N_i}\left( x^{(i)}_k \right)

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    Examples
    --------
    The following example uses PyTorch backend.

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn.pytorch.glob import MaxPooling
    >>>
    >>> g1 = dgl.DGLGraph()
    >>> g1.add_nodes(2)
    >>> g1_node_feats = jnp.ones(2,5)
    >>>
    >>> g2 = dgl.DGLGraph()
    >>> g2.add_nodes(3)
    >>> g2_node_feats = jnp.ones(3,5)
    >>>
    >>> maxpool = MaxPooling()

    Case 1: Input a single graph

    >>> maxpool(g1, g1_node_feats)
        tensor([[1., 1., 1., 1., 1.]])

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = jnp.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> maxpool(batch_g, batch_f)
        tensor([[1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1.]])
    """

    def __call__(self, graph, feat):
        r"""Compute max pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, *)`, where
            :math:`N` is the number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, *)`, where
            :math:`B` refers to the batch size.
        """
        with graph.local_scope():
            graph.ndata['h'] = feat
            readout = max_nodes(graph, 'h')
            return readout


class SortPooling(object):
    r"""

    Description
    -----------
    Apply Sort Pooling (`An End-to-End Deep Learning Architecture for Graph Classification
    <https://www.cse.wustl.edu/~ychen/public/DGCNN.pdf>`__) over the nodes in a graph.

    Parameters
    ----------
    k : int
        The number of nodes to hold for each graph.

    Notes
    -----
        Input: Could be one graph, or a batch of graphs. If using a batch of graphs,
        make sure nodes in all graphs have the same feature size, and concatenate
        nodes' feature together as the input.

    Examples
    --------

    >>> import dgl
    >>> import torch as th
    >>> from dgl.nn.pytorch.glob import SortPooling
    >>>
    >>> g1 = dgl.DGLGraph()
    >>> g1.add_nodes(2)
    >>> g1_node_feats = jnp.ones(2,5)
    >>>
    >>> g2 = dgl.DGLGraph()
    >>> g2.add_nodes(3)
    >>> g2_node_feats = jnp.ones(3,5)
    >>>
    >>> sortpool = SortPooling(k=2)

    Case 1: Input a single graph

    >>> sortpool(g1, g1_node_feats)
        tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])

    Case 2: Input a batch of graphs

    Build a batch of DGL graphs and concatenate all graphs' node features into one tensor.

    >>> batch_g = dgl.batch([g1, g2])
    >>> batch_f = jnp.cat([g1_node_feats, g2_node_feats])
    >>>
    >>> sortpool(batch_g, batch_f)
        tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.],
                [1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])
    """
    def __init__(self, k):
        self.k = k

    def __call__(self, graph, feat):
        r"""

        Compute sort pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, k * D)`, where :math:`B` refers
            to the batch size of input graphs.
        """
        with graph.local_scope():
            # Sort the feature of each node in ascending order.
            feat = feat.sort(axis=-1)
            graph.ndata['h'] = feat
            # Sort nodes according to their last features.
            ret = topk_nodes(graph, 'h', self.k, sortby=-1)[0].reshape((
                -1, self.k * feat.shape[-1]))
            return ret


class GlobalAttentionPooling(nn.Module):
    r"""

    Description
    -----------
    Apply Global Attention Pooling (`Gated Graph Sequence Neural Networks
    <https://arxiv.org/abs/1511.05493.pdf>`__) over the nodes in a graph.

    .. math::
        r^{(i)} = \sum_{k=1}^{N_i}\mathrm{softmax}\left(f_{gate}
        \left(x^{(i)}_k\right)\right) f_{feat}\left(x^{(i)}_k\right)

    Parameters
    ----------
    gate_nn : torch.nn.Module
        A neural network that computes attention scores for each feature.
    feat_nn : torch.nn.Module, optional
        A neural network applied to each feature before combining them with attention
        scores.
    """
    gate_feats: int
    feat_feats: int = 0

    @nn.compact
    def __call__(self, graph, feat):
        r"""

        Compute global attention pooling.

        Parameters
        ----------
        graph : DGLGraph
            A DGLGraph or a batch of DGLGraphs.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)` where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers
            to the batch size.
        """
        with graph.local_scope():
            gate_nn = nn.Dense(self.gate_feats)
            gate = gate_nn(feat)
            assert gate.shape[-1] == 1, "The output of gate_nn should have size 1 at the last axis."

            if self.feat_feats != 0:
                feat_nn = nn.Dense(self.feat_feats)
                feat = feat_nn(feat)

            graph.ndata['gate'] = gate
            gate = softmax_nodes(graph, 'gate')
            graph.ndata.pop('gate')

            graph.ndata['r'] = feat * gate
            readout = sum_nodes(graph, 'r')
            graph.ndata.pop('r')

            return readout


class Set2Set(nn.Module):
    r"""

    Description
    -----------
    For each individual graph in the batch, set2set computes

    .. math::
        q_t &= \mathrm{LSTM} (q^*_{t-1})

        \alpha_{i,t} &= \mathrm{softmax}(x_i \cdot q_t)

        r_t &= \sum_{i=1}^N \alpha_{i,t} x_i

        q^*_t &= q_t \Vert r_t

    for this graph.

    Parameters
    ----------
    input_dim : int
        The size of each input sample.
    n_iters : int
        The number of iterations.
    n_layers : int
        The number of recurrent layers.
    """

    input_dim: int
    n_iters: int
    n_layers: int

    def setup(self):
        self.output_dim = 2 * self.input_dim

    @nn.compact
    def __call__(self, graph, feat):
        r"""
        Compute set2set pooling.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)` where  :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to
            the batch size, and :math:`D` means the size of features.
        """
        with graph.local_scope():
            batch_size = graph.batch_size

            h = (jnp.zeros((self.n_layers, batch_size, self.input_dim)),
                 jnp.zeros((self.n_layers, batch_size, self.input_dim)))

            q_star = jnp.zeros((batch_size, self.output_dim))

            for _ in range(self.n_iters):
                h, q = nn.LSTMCell()(h, jnp.expand_dims(q_star, 0))
                q = q[0].reshape((batch_size, self.input_dim))
                e = (feat * broadcast_nodes(graph, q)).sum(axis=-1, keepdims=True)
                graph.ndata['e'] = e
                alpha = softmax_nodes(graph, 'e')
                graph.ndata['r'] = feat * alpha
                readout = sum_nodes(graph, 'r')
                q_star = jnp.concatenate([q, readout], axis=-1)

            return q_star

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = 'n_iters={n_iters}'
        return summary.format(**self.__dict__)


class MultiHeadAttention(nn.Module):
    r"""Multi-Head Attention block, used in Transformer, Set Transformer and so on."""
    d_model: int
    num_heads: int
    d_head: int
    d_ff: int
    dropouth: float = 0.0
    dropouta: float = 0.0

    def setup(self):
        self.proj_q = nn.Dense(self.num_heads * self.d_head, use_bias=False)
        self.proj_k = nn.Dense(self.num_heads * self.d_head, use_bias=False)
        self.proj_v = nn.Dense(self.num_heads * self.d_head, use_bias=False)
        self.proj_o = nn.Dense(self.d_model, use_bias = False)

        self.norm_in = nn.LayerNorm(self.d_model)
        self.norm_inter = nn.LayerNorm(self.d_model)

    @nn.compact
    def __call__(self, x, mem, lengths_x, lengths_mem):
        """
        Compute multi-head self-attention.

        Parameters
        ----------
        x : torch.Tensor
            The input tensor used to compute queries.
        mem : torch.Tensor
            The memory tensor used to compute keys and values.
        lengths_x : list
            The array of node numbers, used to segment x.
        lengths_mem : list
            The array of node numbers, used to segment mem.
        """
        batch_size = len(lengths_x)
        max_len_x = max(lengths_x)
        max_len_mem = max(lengths_mem)

        queries = self.proj_q(x).reshape((-1, self.num_heads, self.d_head))
        keys = self.proj_k(mem).reshape((-1, self.num_heads, self.d_head))
        values = self.proj_v(mem).reshape((-1, self.num_heads, self.d_head))

        # padding to (B, max_len_x/mem, num_heads, d_head)
        queries = F.pad_packed_tensor(queries, lengths_x, 0)
        keys = F.pad_packed_tensor(keys, lengths_mem, 0)
        values = F.pad_packed_tensor(values, lengths_mem, 0)

        # attention score with shape (B, num_heads, max_len_x, max_len_mem)
        e = jnp.einsum('bxhd,byhd->bhxy', queries, keys)
        # normalize
        e = e / np.sqrt(self.d_head)

        # generate mask
        mask = jnp.zeros((batch_size, max_len_x, max_len_mem))
        for i in range(batch_size):
            mask = jax.ops.index_update(
                mask,
                jax.ops.index[i, :lengths_x[i], :lengths_mem[i]],
                1,
            )
        mask = jnp.expand_dims(mask, 1)

        e = jnp.where(
            mask == 0,
            -float('inf') * jnp.ones_like(e),
            e
        )

        # apply softmax
        alpha = jax.nn.softmax(e, axis=-1)
        # sum of value weighted by alpha
        out = jnp.einsum('bhxy,byhd->bxhd', alpha, values)
        # project to output
        out = self.proj_o(
            out.reshape((batch_size, max_len_x, self.num_heads * self.d_head)))
        # pack tensor
        out = F.pack_padded_tensor(out, lengths_x)

        # intra norm
        x = self.norm_in(x + out)

        # inter norm
        ffn_x = nn.Dense(self.d_model)(
            nn.relu(
                nn.Dropout(self.dropouth)(
                    nn.Dense(self.d_ff)(x)
                )
            )
        )

        x = self.norm_inter(x + ffn_x)

        return x


class SetAttentionBlock(nn.Module):
    r"""SAB block mentioned in Set-Transformer paper."""

    d_model: int
    num_heads: int
    d_head: int
    d_ff: int
    dropouth: float = 0.0
    dropouta: float = 0.0

    def setup(self):
        self.mha = MultiHeadAttention(
            d_model=self.d_model,
            num_heads=self.num_heads,
            d_head=self.d_head,
            d_ff=self.d_ff,
            dropouth=self.dropouth,
            dropouta=self.dropouta,
        )

    def __call__(self, feat, lengths):
        """
        Compute a Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.
        """
        return self.mha(feat, feat, lengths, lengths)

class InducedSetAttentionBlock(nn.Module):
    r"""ISAB block mentioned in Set-Transformer paper."""

    m: int
    d_model: int
    num_heads: int
    d_head: int
    d_ff: int
    dropouth: int = 0.0
    dropouta: int = 0.0

    def setup(self):
        self.inducing_points = self.param(
            "inducing_points",
            nn.initializers.xavier_uniform(),
            ((self.m, self.d_model)),
        )

        self.mha = [
            MultiHeadAttention(self.d_model, self.num_heads, self.d_head, self.d_ff,
                               dropouth=self.dropouth, dropouta=self.dropouta) for _ in range(2)]

    def __call__(self, feat, lengths):
        """
        Compute an Induced Set Attention Block.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        batch_size = len(lengths)
        query = self.inducing_points.repeat(batch_size, axis=0)
        memory = self.mha[0](query, feat, [self.m] * batch_size, lengths)
        return self.mha[1](feat, memory, lengths, [self.m] * batch_size)

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        shape_str = '({}, {})'.format(self.inducing_points.shape[0], self.inducing_points.shape[1])
        return 'InducedVector: ' + shape_str


class PMALayer(nn.Module):
    r"""Pooling by Multihead Attention, used in the Decoder Module of Set Transformer."""
    k: int
    d_model: int
    num_heads: int
    d_head: int
    d_ff: int
    dropouth: float = 0.0
    dropouta: float = 0.0

    def setup(self):
        self.seed_vectors = self.param(
            "seed_vectors",
            nn.initializers.xavier_uniform(),
            (self.k, self.d_model),
        )

        self.mha = MultiHeadAttention(self.d_model, self.num_heads, self.d_head, self.d_ff,
                                      dropouth=self.dropouth, dropouta=self.dropouta)

    @nn.compact
    def __call__(self, feat, lengths):
        """
        Compute Pooling by Multihead Attention.

        Parameters
        ----------
        feat : torch.Tensor
            The input feature.
        lengths : list
            The array of node numbers, used to segment feat tensor.

        Returns
        -------
        torch.Tensor
            The output feature
        """
        batch_size = len(lengths)
        query = self.seed_vectors.repeat(batch_size, axis=0)

        ffn_feat = nn.Dense(self.d_model)(
            nn.relu(
                nn.Dropout(self.dropouth)(
                    nn.Dense(self.d_ff)(feat)
                )
            )
        )

        return self.mha(query, ffn_feat, [self.k] * batch_size, lengths)

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        shape_str = '({}, {})'.format(self.seed_vectors.shape[0], self.seed_vectors.shape[1])
        return 'SeedVector: ' + shape_str


class SetTransformerEncoder(nn.Module):
    r"""

    Description
    -----------
    The Encoder module in `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/pdf/1810.00825.pdf>`__.

    Parameters
    ----------
    d_model : int
        The hidden size of the model.
    n_heads : int
        The number of heads.
    d_head : int
        The hidden size of each head.
    d_ff : int
        The kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        The number of layers.
    block_type : str
        Building block type: 'sab' (Set Attention Block) or 'isab' (Induced
        Set Attention Block).
    m : int or None
        The number of induced vectors in ISAB Block. Set to None if block type
        is 'sab'.
    dropouth : float
        The dropout rate of each sublayer.
    dropouta : float
        The dropout rate of attention heads.
    """

    d_model: int
    n_heads: int
    d_head: int
    d_ff: int
    n_layers: int
    block_type: str="sab"
    from typing import Union
    m: Union[None, int] = None
    dropouth: float = 0.0
    dropouta: float = 0.0

    def setup(self):
        print(self.m)
        layers = []
        if self.block_type == 'isab' and self.m is None:
            raise KeyError('The number of inducing points is not specified in ISAB block.')

        for _ in range(self.n_layers):
            if self.block_type == 'sab':
                layers.append(
                    SetAttentionBlock(self.d_model, self.n_heads, self.d_head, self.d_ff,
                                      dropouth=self.dropouth, dropouta=self.dropouta))
            elif self.block_type == 'isab':
                layers.append(
                    InducedSetAttentionBlock(self.m, self.d_model, self.n_heads, self.d_head, self.d_ff,
                                             dropouth=self.dropouth, dropouta=self.dropouta))
            else:
                raise KeyError("Unrecognized block type {}: we only support sab/isab")

        self.layers = layers

    def __call__(self, graph, feat):
        """
        Compute the Encoder part of Set Transformer.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(N, D)`.
        """
        lengths = graph.batch_num_nodes()
        for layer in self.layers:
            print(layer)
            feat = layer(feat, lengths)
        return feat


class SetTransformerDecoder(nn.Module):
    r"""

    Description
    -----------
    The Decoder module in `Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks <https://arxiv.org/pdf/1810.00825.pdf>`__.

    Parameters
    ----------
    d_model : int
        Hidden size of the model.
    num_heads : int
        The number of heads.
    d_head : int
        Hidden size of each head.
    d_ff : int
        Kernel size in FFN (Positionwise Feed-Forward Network) layer.
    n_layers : int
        The number of layers.
    k : int
        The number of seed vectors in PMA (Pooling by Multihead Attention) layer.
    dropouth : float
        Dropout rate of each sublayer.
    dropouta : float
        Dropout rate of attention heads.
    """
    d_model: int
    num_heads: int
    d_head: int
    d_ff: int
    n_layers: int
    k: int
    dropouth: float = 0.0
    dropouta: float = 0.0

    def setup(self):
        self.pma = PMALayer(self.k, self.d_model, self.num_heads, self.d_head, self.d_ff,
                            dropouth=self.dropouth, dropouta=self.dropouta)
        layers = []
        for _ in range(self.n_layers):
            layers.append(
                SetAttentionBlock(self.d_model, self.num_heads, self.d_head, self.d_ff,
                                  dropouth=self.dropouth, dropouta=self.dropouta))

        self.layers = layers

    def __call__(self, graph, feat):
        """
        Compute the decoder part of Set Transformer.

        Parameters
        ----------
        graph : DGLGraph
            The input graph.
        feat : torch.Tensor
            The input feature with shape :math:`(N, D)`, where :math:`N` is the
            number of nodes in the graph, and :math:`D` means the size of features.

        Returns
        -------
        torch.Tensor
            The output feature with shape :math:`(B, D)`, where :math:`B` refers to
            the batch size.
        """
        len_pma = graph.batch_num_nodes()
        len_sab = [self.k] * graph.batch_size
        feat = self.pma(feat, len_pma)
        for layer in self.layers:
            feat = layer(feat, len_sab)
        return feat.reshape((graph.batch_size, self.k * self.d_model))


class WeightAndSum(nn.Module):
    """Compute importance weights for atoms and perform a weighted sum.

    Parameters
    ----------
    in_feats : int
        Input atom feature size
    """
    def __init__(self, in_feats):
        super(WeightAndSum, self).__init__()
        self.in_feats = in_feats
        self.atom_weighting = nn.Sequential(
            nn.Linear(in_feats, 1),
            nn.Sigmoid()
        )

    def __call__(self, g, feats):
        """Compute molecule representations out of atom representations

        Parameters
        ----------
        g : DGLGraph
            DGLGraph with batch size B for processing multiple molecules in parallel
        feats : FloatTensor of shape (N, self.in_feats)
            Representations for all atoms in the molecules
            * N is the total number of atoms in all molecules

        Returns
        -------
        FloatTensor of shape (B, self.in_feats)
            Representations for B molecules
        """
        with g.local_scope():
            g.ndata['h'] = feats
            g.ndata['w'] = self.atom_weighting(g.ndata['h'])
            h_g_sum = sum_nodes(g, 'h', 'w')

        return h_g_sum
