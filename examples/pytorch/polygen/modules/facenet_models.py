from .attention import *
from .layers import *
from .embedding import *
from dataset import *
import threading
import torch as th
import dgl.nn as dglnn
import dgl.function as fn
import torch.nn as nn
import torch.nn.init as INIT

class Encoder(nn.Module):
    def __init__(self, layer, N):
        '''
        args:
            N: number of blocks.
        '''
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
            x, z = nodes.data['x'], nodes.data['z']
            o = layer.self_attn.get_o(z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            # We apply norm at the begining for each block. Then output of transformer need an additional norm.
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Decoder(nn.Module):
    def __init__(self, layer, N):
        '''
        args:
            N: number of blocks.
        '''
        super(Decoder, self).__init__()
        self.N = N
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def pre_func(self, i, fields='qkv'):
        layer = self.layers[i]
        def func(nodes):
            x = nodes.data['x']
            norm_x = layer.sublayer[0].norm(x)
            return layer.self_attn.get(norm_x, fields)
        return func

    def post_func(self, i):
        layer = self.layers[i]
        def func(nodes):
            x, z = nodes.data['x'], nodes.data['z']
            o = layer.self_attn.get_o(z)
            x = x + layer.sublayer[0].dropout(o)
            x = layer.sublayer[1](x, layer.feed_forward)
            # We apply norm at the begining for each block. Then output of transformer need an additional norm.
            return {'x': x if i < self.N - 1 else self.norm(x)}
        return func

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, vert_val_embed, vert_pos_enc, face_pos_enc, vert_idx_in_face_enc, generator, h, d_k):
        '''
        args:
            encoder: encoder module
            decoder: decoder module
            vert_val_embed: vertex value embedding module
            vert_pos_enc: vertex position encoding module
            face_pos_enc: face position encoding module
            vert_idx_in_face_enc: vertex idx in face encoding module
            h: number of heads
            d_k: dim per head
        '''
        super(Transformer, self).__init__()
        self.encoder,  self.decoder = encoder, decoder
        self.vert_val_embed = vert_val_embed
        self.vert_pos_enc = vert_pos_enc
        self.face_pos_enc = face_pos_enc
        self.vert_idx_in_face_enc = vert_idx_in_face_enc
        self.h, self.d_k = h, d_k
        self.att_weight_map = None
        self.generator = generator

    def propagate_attention(self, g, eids):
        # Compute attention score
        g.apply_edges(fn.u_dot_v('k', 'q', 'score'), eids)
        # Softmax
        g.edata['softmax_score'] = th.zeros_like(g.edata['score'].unsqueeze(-1))
        g.edata['softmax_score'][eids] = dglnn.edge_softmax(g, g.edata['score'][eids], eids).unsqueeze(-1)
        # sum up v
        g.send_and_recv(eids,
                        [fn.u_mul_e('v', 'softmax_score', 'z')],
                        [fn.sum('z', 'z')])

    def update_graph(self, g, eids, pre_pairs, post_pairs):
        "Update the node states and edge states of the graph."

        # Pre-compute queries and key-value pairs.
        for pre_func, nids in pre_pairs:
            g.apply_nodes(pre_func, nids)
        self.propagate_attention(g, eids)
        # Further calculation after attention mechanism
        for post_func, nids in post_pairs:
            g.apply_nodes(post_func, nids)

    def forward(self, graph):
        g = graph.g
        nids, eids = graph.nids, graph.eids
        # Embed all vertex
        vert_val_embed = self.vert_val_embed(graph.src[0])
        vert_pos_enc = self.vert_pos_enc(graph.src[1])
        g.nodes[nids['enc']].data['x'] = vert_val_embed + vert_pos_enc
        g.nodes[nids['enc']].data['idx'] = graph.src[1].float()
        
        # Run encoder
        for i in range(self.encoder.N):
            pre_func = self.encoder.pre_func(i, 'qkv')
            post_func = self.encoder.post_func(i)
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])
        
        # Indexing encoder results
        tgt_embed = g.nodes[nids['enc']].data['x'].index_select(0, graph.tgt[0])
        
        # Embed face position
        face_pos_enc = self.face_pos_enc(graph.tgt[1]//3)
        vert_idx_in_face_enc = self.vert_idx_in_face_enc(graph.tgt[1]%3)
        g.nodes[nids['dec']].data['x'] = tgt_embed + face_pos_enc + vert_idx_in_face_enc
        g.nodes[nids['dec']].data['idx'] = graph.tgt[1].float()
        
        # Run decoder
        for i in range(self.decoder.N):
            pre_func = self.decoder.pre_func(i, 'qkv')
            post_func = self.decoder.post_func(i)
            nodes, edges = nids['dec'], eids['dd']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])
        
        nodes, edges = nids['dec'], eids['ed']

        # Pointer network
        # log(softmax) is not numerical stable.
        # We will only output the dot_product result and use CrossEntropyLoss
        # which combines log_softmax and nllloss
        g.apply_edges(fn.u_dot_v('x', 'x', 'pointer_score'), edges)
        pointer_score_flat = g.edata['pointer_score'][edges]
        src_lens, tgt_lens = graph.src_lens, graph.tgt_lens
        pointer_score_padded = -1e8 * th.ones([np.sum(tgt_lens), FaceDataset.MAX_VERT_LENGTH], dtype=th.float, device=tgt_embed.device)
        # pad softmax res
        cur_idx = 0
        cur_tgt_head = 0
        for n_dec, n_enc in zip(tgt_lens, src_lens):
            pointer_score_padded[cur_tgt_head:cur_tgt_head+n_dec, :n_enc] = th.transpose(pointer_score_flat[cur_idx:cur_idx+n_enc*n_dec].reshape([n_enc, n_dec]),0,1)
            cur_idx += n_enc * n_dec
            cur_tgt_head += n_dec
        return pointer_score_padded


    def infer(self, graph, max_len, eos_id):
        '''
        This function implements Beam Search in DGL, which is required in inference phase.
        args:
            graph: a `Graph` object defined in `dgl.contrib.transformer.graph`.
            max_len: the maximum length of decoding.
            eos_id: the index of end-of-sequence symbol.
        return:
            ret: a list of index array correspond to the input sequence specified by `graph``.
        '''
        g = graph.g
        N, E = graph.n_nodes, graph.n_edges
        nids, eids = graph.nids, graph.eids

        # Embed all vertex
        vert_val_embed = self.vert_val_embed(graph.src[0])
        vert_pos_enc = self.vert_pos_enc(graph.src[1])
        g.nodes[nids['enc']].data['x'] = self.vert_pos_enc.dropout(vert_val_embed + vert_pos_enc)
        g.nodes[nids['enc']].data['idx'] = graph.src[1]

        # init mask
        device = next(self.parameters()).device
        g.ndata['mask'] = th.zeros(N, dtype=th.uint8, device=device).bool()

        # encode
        for i in range(self.encoder.N):
            pre_func = self.encoder.pre_func(i, 'qkv')
            post_func = self.encoder.post_func(i)
            nodes, edges = nids['enc'], eids['ee']
            self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])

        # decode
        log_prob = None
        y = graph.tgt[0]

        # Indexing encoder results
        tgt_embed = g.nodes[nids['enc']].data['x'].index_select(0, graph.tgt[0])
        
        # Indexing result
        face_pos_enc = self.face_pos_enc(graph.tgt[1]//3)
        vert_idx_in_face_enc = self.vert_idx_in_face_enc(graph.tgt[1]%3)
        g.nodes[nids['dec']].data['idx'] = graph.tgt[1].float()

        for step in range(1, max_len):
            y = y.view(-1)
            tgt_embed = g.nodes[nids['enc']].data['x'].index_select(0, y)
            g.nodes[nids['dec']].data['x'] = tgt_embed + face_pos_enc + vert_idx_in_face_enc
            edges_ed = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'] , eids['ed'])
            edges_dd = g.filter_edges(lambda e: (e.dst['pos'] < step) & ~e.dst['mask'], eids['dd'])
            nodes_d = g.filter_nodes(lambda v: (v.data['pos'] < step) & ~v.data['mask'], nids['dec'])
            for i in range(self.decoder.N):
                pre_func, post_func = self.decoder.pre_func(i, 'qkv'), self.decoder.post_func(i)
                nodes, edges = nodes_d, edges_dd
                self.update_graph(g, edges, [(pre_func, nodes)], [(post_func, nodes)])

            nodes, edges = nids['dec'], eids['ed']

            # Pointer network
            # log(softmax) is not numerical stable.
            # We will only output the dot_product result and use CrossEntropyLoss
            # which combines log_softmax and nllloss
            g.apply_edges(fn.u_dot_v('x', 'x', 'pointer_score'), edges)
            pointer_score_flat = g.edata['pointer_score'][edges]
            src_lens, tgt_lens = graph.src_lens, graph.tgt_lens
            pointer_score_padded = -1e8 * th.ones([np.sum(tgt_lens), FaceDataset.MAX_VERT_LENGTH], dtype=th.float, device=tgt_embed.device)
            # pad softmax res
            cur_idx = 0
            cur_tgt_head = 0
            for n_dec, n_enc in zip(tgt_lens, src_lens):
                pointer_score_padded[cur_tgt_head:cur_tgt_head+n_dec, :n_enc] = th.transpose(pointer_score_flat[cur_idx:cur_idx+n_enc*n_dec].reshape([n_enc, n_dec]),0,1)
                cur_idx += n_enc * n_dec
                cur_tgt_head += n_dec

            batch_size = pointer_score_padded.shape[0] // tgt_lens[0]
            frontiers = [i*tgt_lens[0]+step for i in range(batch_size)]
            out = pointer_score_padded[frontiers].max(dim=-1)

            y = y.view(batch_size, -1)
            for i in range(batch_size):
                y[i, step] = out[i]
                eos[i] = eos[i] | (out[i] == eos_id).bool()
            
            if eos.all():
                break
            else:
                # Mask out all the nodes from the sample already met eos
                print (eos, g.ndata['mask'][nids['dec']].shape, max_len)
                g.ndata['mask'][nids['dec']] = eos.unsqueeze(-1).repeat(1, max_len).view(-1).to(device)
        return y.view(batch_size, -1).tolist()


    def _register_att_map(self, g, enc_ids, dec_ids):
        self.att_weight_map = [
            get_attention_map(g, enc_ids, enc_ids, self.h),
            get_attention_map(g, enc_ids, dec_ids, self.h),
            get_attention_map(g, dec_ids, dec_ids, self.h),
        ]


def make_face_model(N=12, dim_model=256, dim_ff=1024, h=8, dropout=0.1,, infer = False, cumulative_p=0.9):
    '''
    args:
        N: transformer block number.
        dim_model: hidden layer dimention of transformer.
        dim_ff: feedforward layer dimention.
        h: number of head for multihead attention
        dropout: dropout rate
        infer: mode
        cumulative_p: cumulative threshold for nucleus sampling
        max_generator: use max to infer
    '''
    c = copy.deepcopy
    attn = MultiHeadAttention(h, dim_model)
    ff = PositionwiseFeedForward(dim_model, dim_ff)
    # Embeddings
    vert_pos_enc = Embeddings(FaceDataset.MAX_VERT_LENGTH, dim_model)
    face_pos_enc = Embeddings(FaceDataset.MAX_FACE_LENGTH, dim_model)
    vert_idx_in_face_enc = Embeddings(3, dim_model)
    vert_val_vocab = FaceDataset.COORD_BIN + 3
    vert_val_embed = VertCoordJointEmbeddings(vert_val_vocab, dim_model)

    encoder = Encoder(EncoderLayer(dim_model, c(attn), c(ff), dropout), N)
    decoder = Decoder(DecoderLayer(dim_model, c(attn), None, c(ff), dropout), N)

    if infer:
        if cumulative_p > 0:
            generator = NucleusSamplingGenerator(dim_model, tgt_vocab=tgt_vocab, cumulative_p=cumulative_p)
    else:
        generator = None

    model = Transformer(
        encoder, decoder, vert_val_embed, vert_pos_enc, face_pos_enc, vert_idx_in_face_enc, generator, h, dim_model // h)
    # xavier init
    for p in model.parameters():
        if p.dim() > 1:
            INIT.xavier_uniform_(p)
    return model