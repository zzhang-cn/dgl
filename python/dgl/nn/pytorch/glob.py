"""Torch modules for graph global pooling."""
# pylint: disable= no-member, arguments-differ
import torch as th
import torch.nn as nn
import numpy as np

from torch.nn import init
from .softmax import edge_softmax
from ... import function as fn, BatchedDGLGraph
from ...utils import get_ndata_name, get_edata_name
from ...batched_graph import sum_nodes, mean_nodes, max_nodes, broadcast_nodes, softmax_nodes, topk_nodes


class SumPooling(nn.Module):
    r"""Apply sum pooling over the graph.
    """
    _feat_name = '_gpool_feat'
    def __init__(self):
        super(SumPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = sum_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class AvgPooling(nn.Module):
    r"""Apply average pooling over the graph.
    """
    _feat_name = '_gpool_avg'
    def __init__(self):
        super(AvgPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = mean_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class MaxPooling(nn.Module):
    r"""Apply max pooling over the graph.
    """
    _feat_name = '_gpool_max'
    def __init__(self):
        super(MaxPooling, self).__init__()

    def forward(self, feat, graph):
        _feat_name = get_ndata_name(graph, self._feat_name)
        graph.ndata[_feat_name] = feat
        readout = max_nodes(graph, _feat_name)
        graph.ndata.pop(_feat_name)
        return readout


class SortPooling(nn.Module):
    r"""Apply sort pooling (f"An End-to-End Deep Learning Architecture
    for Graph Classification") over the graph.
    """
    _feat_name = '_gpool_sort'
    def __init__(self, k):
        super(SortPooling, self).__init__()
        self.k = k

    def forward(self, feat, graph):
        # Sort the feature of each node in ascending order.
        feat, _ = feat.sort(dim=-1)
        graph.ndata[self._feat_name] = feat
        # Sort nodes according to the their last features.
        ret = topk_nodes(graph, self._feat_name, self.k).view(-1, self.k * feat.shape[-1])
        g.ndata.pop(self._feat_name)
        return ret


class GlobAttnPooling(nn.Module):
    r"""Apply global attention pooling over the graph.
    """
    _gate_name = '_gpool_attn_gate'
    _readout_name = '_gpool_attn_readout'
    def __init__(self, gate_nn, nn=None):
        super(GlobAttnPooling, self).__init__()
        self.gate_nn = gate_nn
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.gate_nn:
            init.xavier_uniform_(p)
        for p in self.nn:
            init.xavier_uniform_(p)

    def forward(self, feat, graph):
        feat = feat.unsqueeze(-1) if feat.dim() == 1 else feat
        gate = self.gate_nn(feat)
        feat = self.nn(feat) if self.nn else feat

        feat_name = get_ndata_name(graph, self.gate_name)
        graph.ndata[feat_name] = gate
        gate = softmax_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        feat_name = get_ndata_name(graph, self.readout_name)
        graph.ndata[feat_name] = feat * gate
        readout = sum_nodes(graph, feat_name)
        graph.ndata.pop(feat_name)

        return readout


class Set2Set(nn.Module):
    r"""Apply Set2Set (f"Order Matters: Sequence to sequence for sets") over the graph.
    """
    _score_name = '_gpool_s2s_score'
    _readout_name = '_gpool_s2s_readout'
    def __init__(self, input_dim, n_iters, n_layers):
        super(Set2Set, self).__init__()
        self.input_dim = input_dim
        self.output_dim = 2 * input_dim
        self.n_iters = n_iters
        self.n_layers= n_layers
        self.lstm = th.nn.LSTM(self.output_dim, self.input_dim, n_layers)
        self.reset_parameters()

    def reset_parameters(self):
        for p in self.lstm.parameters():
            init.xavier_uniform_(p)

    def forward(self, feat, graph):
        batch_size = 1
        if isinstance(graph, BatchedDGLGraph):
            batch_size = graph.batch_size

        h = (feat.new_zeros((self.n_layers, batch_size, self.input_dim)),
             feat.new_zeros((self.n_layers, batch_size, self.input_dim)))
        q_star = feat.new_zeros(batch_size, self.output_dim)

        for i in range(self.n_iters):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, self.input_dim)

            score = (feat * broadcast_nodes(graph, q)).sum(dim=-1, keepdim=True)
            feat_name = get_ndata_name(graph, self._score_name)
            graph.ndata[feat_name] = score
            score = softmax_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            feat_name = get_ndata_name(graph, self._readout_name)
            graph.ndata[feat_name] = feat * score
            readout = sum_nodes(graph, feat_name)
            graph.ndata.pop(feat_name)

            q_star = th.cat([q, readout], dim=-1)

        return q_star

    def extra_repr(self):
        """Set the extra representation of the module.
        which will come into effect when printing the model.
        """
        summary = 'input_dim={input_dim}, out_dim={out_dim}' +\
            'n_iters={n_iters}, n_layers={n_layers}'
        return summary.format(**self.__dict__)


class MultiHeadAttention(nn.Module):
    _query_name = '_gpool_mha_query'
    _key_name = '_gpool_mha_key'
    _value_name = '_gpool_mha_value'
    _score_name = '_gpool_mha_score'
    _att_name = '_gpool_mha_att'
    _out_name = '_gpool_mha_out'
    _feat_name = '_gpool_mha_feat'
    def __init__(self, d_model, num_heads, d_head, d_ff, dropouth=0., dropouta=0.):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_head
        self.d_ff = d_ff
        self.W_q = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.W_k = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.W_v = nn.Linear(d_model, num_heads * d_head, bias=False)
        self.W_o = nn.Linear(num_heads * d_head, d_model, bias=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropouth),
            nn.Linear(d_ff, d_model)
        )
        self.droph = nn.Dropout(dropouth)
        self.dropa = nn.Dropout(dropouta)
        self.norm_in = nn.LayerNorm(d_model)
        self.norm_inter = nn.LayerNorm(d_model)

    def forward(self, graph, q_feat, kv_feat, q_nids, kv_nids):
        feat_name = get_ndata_name(graph, self._feat_name)
        query_name = get_ndata_name(graph, self._query_name)
        key_name = get_ndata_name(graph, self._key_name)
        value_name = get_ndata_name(graph, self._value_name)
        score_name = get_ndata_name(graph, self._score_name)
        att_name = get_ndata_name(graph, self._att_name)
        out_name = get_ndata_name(graph, self._out_name)

        # Copy q_feat and kv_feat to graph data frame
        graph.ndata[q_nids][feat_name] = q_feat
        graph.ndata[kv_nids][feat_name] = kv_feat

        # Compute queries, keys and values.
        graph.nodes[q_nids][query_name] =\
            self.W_q(graph.nodes[q_nids][feat_name]).view(-1, self.num_heads, self.d_head)
        graph.nodes[kv_nids][key_name] =\
            self.W_k(graph.nodes[kv_nids][feat_name]).view(-1, self.num_heads, self.d_head)
        graph.nodes[kv_nids][value_name] =\
            self.W_v(graph.nodes[kv_nids][feat_name]).view(-1, self.num_heads, self.d_head)

        # Free node features.
        graph.ndata.pop(feat_name)

        # Compute attention score.
        graph.apply_edges(fn.u_mul_v(key_name, query_name, score_name))
        e = graph.edata.pop(score_name).sum(dim=-1, keepdim=True) / np.sqrt(self.d_head) # Attention & Free score field.
        graph.edata[att_name] = self.dropa(edge_softmax(graph, e))
        graph.pull(q_nids,
                   fn.u_mul_e(value_name, att_name, 'm'),
                   fn.sum('m', out_name))
        sa = self.W_o(graph.nodes[q_nids][out_name].view(-1, self.num_heads * self.d_head))
        feat = self.norm_in(q_feat + sa)

        # Free queries, keys, values, outputs and attention weights.
        graph.ndata.pop(query_name)
        graph.ndata.pop(key_name)
        graph.ndata.pop(value_name)
        graph.ndata.pop(out_name)
        graph.edata.pop(att_name)

        # Position-wise Feed Forward Network
        feat = self.norm_inter(feat + self.ffn(feat))

        return feat


class SetAttentionBlock(nn.Module):
    def __init__(self):
        super(SetAttentionBlock, self).__init__()
        pass

class InducedSetAttentionBlock(nn.Module):
    def __init__(self):
        super(InducedSetAttentionBlock, self).__init__()
        pass

class MHAPooling(nn.Module):
    def __init__(self):
        super(MHAPooling, self).__init__()
        pass

class STEncoder(nn.Module):
    def __init__(self):
        super(STEncoder, self).__init__()
        pass

class STDecoder(nn.Module):
    def __init__(self):
        super(STDecoder, self).__init__()
        pass

class SetTransformer(nn.Module):
    r"""Apply Set Transformer(f""Set Transformer: A Framework for Attention-based
    Permutation-Invariant Neural Networks") over the graph.
    """
    def __init__(self):
        pass

    def forward(self, feat, graph):
        pass


