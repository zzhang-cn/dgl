from .layers import *
from .functions import *
from .viz import *
from .config import *
from .act import *
import torch as th
import threading

class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields=fields)
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[0].dropout(o)
            return {'x': x}
        return func

    def ff_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x = layer.sublayer[1](nodes.data['x'], layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv', l=0):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            if fields == 'kv':
                norm_x = x # In enc-dec attention, x has already been normalized.
            else:
                norm_x = layer.sublayer[l].norm(x)
            return layer.self_attn.get(norm_x, fields)
        return func

    def post_func(self, i, l=0):
        layer = self.layers[i]
        def func(nodes):
            x, wv, z = nodes.data['x'], nodes.data['wv'], nodes.data['z']
            o = layer.self_attn.get_o(wv / z)
            x = x + layer.sublayer[l].dropout(o)
            return {'x': x}
        return func

    def ff_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x = layer.sublayer[2](nodes.data['x'], layer.feed_forward)
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

lock = threading.Lock()

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, src_embed, tgt_embed, pos_enc, generator, h, d_k):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.pos_enc = pos_enc
        self.generator = generator
        self.h = h
        self.d_k = d_k
        self.att_weight_map = None

    def _register_att_map(self, g, enc_ids, dec_ids):
        self.att_weight_map = [
            get_attention_map(g, enc_ids, enc_ids),
            get_attention_map(g, enc_ids, dec_ids),
            get_attention_map(g, dec_ids, dec_ids),
        ]

    def update_graph(self, g, eids, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."

        # Pre-compute queries and key-value pairs.
        for pre_func, nids in pre_pairs:
            g.apply_nodes(pre_func, nids)

        # Compute attention score
        g.apply_edges(src_dot_dst('k', 'q', 'score'), eids)
        g.apply_edges(scaled_exp('score', np.sqrt(self.d_k)))

        # Update node state
        g.send_and_recv(eids,
                        [fn.src_mul_edge('v', 'score', 'v'), fn.copy_edge('score', 'score') ],
                        [fn.sum('v', 'wv'), fn.sum('score', 'z')])

        for post_func, nids in post_pairs:
            g.apply_nodes(post_func, nids)

    def forward(self, graph):
        g = graph.g
        N = graph.n_nodes
        E = graph.n_edges
        h = self.h
        d_k = self.d_k
        nids = graph.nids
        eids = graph.eids

        # embed
        src_embed, src_pos = self.src_embed(graph.src[0]), self.pos_enc(graph.src[1])
        tgt_embed, tgt_pos = self.tgt_embed(graph.tgt[0]), self.pos_enc(graph.tgt[1])
        g.nodes[nids['enc']].data['x'] = self.pos_enc.dropout(src_embed + src_pos)
        g.nodes[nids['dec']].data['x'] = self.pos_enc.dropout(tgt_embed + tgt_pos)

        for i in range(self.encoder.N):
            pre_func, post_func, ff = self.encoder.pre_func(i, 'qkv'), self.encoder.post_func(i), self.encoder.ff_func(i)
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes), (ff, nodes)])

        for i in range(self.decoder.N):
            pre_func, post_func = self.decoder.pre_func(i, 'qkv'), self.decoder.post_func(i)
            nodes, edges = nids['dec'], eids['dd']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])
            pre_q, pre_kv, post_func, ff = self.decoder.pre_func(i, 'q', 1), self.decoder.pre_func(i, 'kv', 1), self.decoder.post_func(i, 1), self.decoder.ff_func(i)
            nodes_e, nodes_d, edges = nids['enc'], nids['dec'], eids['ed']
            self.update_graph(g, edges, [(pre_q, nodes_d), (pre_kv, nodes_e)], [(post_func, nodes_d), (ff, nodes_d)])

        # visualize attention
        with lock:
            if self.att_weight_map is None:
                self._register_att_map(g, graph.nid_arr['enc'][VIZ_IDX], graph.nid_arr['dec'][VIZ_IDX])

        return self.generator(g.ndata['x'][nids['dec']])

    def infer(self, graph, max_len, eos_id, k):
        '''
        This function implements Beam Search in DGL, which is required in inference phase.
        args:
            graph: a `Graph` object defined in `dgl.contrib.transformer.graph`.
            max_len: the maximum length of decoding.
            eos_id: the index of end-of-sequence symbol.
            k: beam size

        return:
            ret: a list of index array correspond to the input sequence specified by `graph``.
        '''
        g = graph.g
        N = graph.n_nodes
        E = graph.n_edges
        h = self.h
        d_k = self.d_k
        nids = graph.nids
        eids = graph.eids

        # embed & pos
        src_embed = self.src_embed(graph.src[0])
        src_pos = self.pos_enc(graph.src[1])
        g.nodes[nids['enc']].data['pos'] = graph.src[1]
        g.nodes[nids['enc']].data['x'] = self.pos_enc.dropout(src_embed + src_pos)
        tgt_pos = self.pos_enc(graph.tgt[1])
        g.nodes[nids['dec']].data['pos'] = graph.tgt[1]

        # init mask
        device = next(self.parameters()).device
        g.ndata['mask'] = th.zeros(N, dtype=th.uint8, device=device)

        # encode
        for i in range(self.encoder.N):
            pre_func, post_func, ff = self.encoder.pre_func(i, 'qkv'), self.encoder.post_func(i), self.encoder.ff_func(i)
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes), (ff, nodes)])

        # decode
        log_prob = None
        y = graph.tgt[0]
        for step in range(1, max_len):
            y = y.view(-1)
            tgt_embed = self.tgt_embed(y)
            g.ndata['x'][nids['dec']] = self.pos_enc.dropout(tgt_embed + tgt_pos)
            edges_ed = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'] , eids['ed'])
            edges_dd = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'], eids['dd'])
            nodes_d = g.filter_nodes(lambda v: (v.data['pos'] < step) & ~v.data['mask'], nids['dec'])
            for i in range(self.decoder.N):
                pre_func, post_func = self.decoder.pre_func(i, 'qkv'), self.decoder.post_func(i)
                nodes, edges = nodes_d, edges_dd
                self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])
                pre_q, pre_kv, post_func, ff = self.decoder.pre_func(i, 'q', 1), self.decoder.pre_func(i, 'kv', 1), self.decoder.post_func(i, 1), self.decoder.ff_func(i)
                nodes_e, nodes_d, edges = nids['enc'], nodes_d, edges_ed
                self.update_graph(g, edges, [(pre_q, nodes_d), (pre_kv, nodes_e)], [(post_func, nodes_d), (ff, nodes_d)])

            frontiers = g.filter_nodes(lambda v: v.data['pos'] == step - 1, nids['dec'])
            out = self.generator(g.ndata['x'][frontiers])
            batch_size = frontiers.shape[0] // k
            vocab_size = out.shape[-1]
            if log_prob is None:
                log_prob, pos = out.view(batch_size, k, -1)[:, 0, :].topk(k, dim=-1)
                eos = th.zeros(batch_size).byte()
            else:
                log_prob, pos = (out.view(batch_size, k, -1) + log_prob.unsqueeze(-1)).view(batch_size, -1).topk(k, dim=-1)

            _y = y.view(batch_size * k, -1)
            y = th.zeros_like(_y)
            for i in range(batch_size):
                if not eos[i]:
                    for j in range(k):
                        _j = pos[i, j].item() // vocab_size
                        token = pos[i, j].item() % vocab_size
                        y[i*k+j,:] = _y[i*k+_j, :]
                        y[i*k+j, step] = token
                        if j == 0:
                            eos[i] = eos[i] | (token == eos_id)
                else:
                    y[i*k:(i+1)*k, :] = _y[i*k:(i+1)*k, :]

            if eos.all():
                break
            else:
                g.ndata['mask'][nids['dec']] = eos.unsqueeze(-1).repeat(1, k * max_len).view(-1).to(device)
        return y.view(batch_size, k, -1)[:, 0, :].tolist()


from .attention import *
from .embedding import *
import torch.nn.init as INIT

def make_model(src_vocab, tgt_vocab, N=6,
                   dim_model=512, dim_ff=2048, h=8, dropout=0.1, universal=False):
    if universal:
        return make_universal_model(src_vocab, tgt_vocab, dim_model, dim_ff, h, dropout)
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model, dropout)
    ff = PositionwiseFeedForward(dim_model, dim_ff, dropout)
    pos_enc = PositionalEncoding(dim_model, dropout)

    encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
    decoder = Decoder(DecoderLayer(dim_model, c(attn), c(attn), c(ff), dropout), N)
    src_embed = Embeddings(src_vocab, dim_model)
    tgt_embed = Embeddings(tgt_vocab, dim_model)
    generator = Generator(dim_model, tgt_vocab)
    model = Transformer(
        encoder, decoder, src_embed, tgt_embed, pos_enc, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model
