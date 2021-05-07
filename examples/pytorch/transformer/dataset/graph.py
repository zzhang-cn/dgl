import dgl
import torch as th
import numpy as np
import itertools
import time
from collections import *

Graph = namedtuple('Graph',
                   ['g', 'src', 'tgt', 'tgt_y', 'nids', 'eids', 'nid_arr', 'n_nodes', 'n_edges', 'n_tokens'])

class GraphPool:
    "Create a graph pool in advance to accelerate graph building phase in Transformer."
    def __init__(self, n=50, m=50):
        '''
        args:
            n: maximum length of input sequence.
            m: maximum length of output sequence.
        '''
        print('start creating graph pool...')
        tic = time.time()
        self.n, self.m = n, m
        g_pool = [[dgl.graph(([], [])) for _ in range(m)] for _ in range(n)]
        num_edges = {
            'ee': np.zeros((n, m)).astype(int),
            'ed': np.zeros((n, m)).astype(int),
            'dd': np.zeros((n, m)).astype(int)
        }
        for i, j in itertools.product(range(n), range(m)):
            src_length = i + 1
            tgt_length = j + 1

            g_pool[i][j].add_nodes(src_length + tgt_length)
            enc_nodes = th.arange(src_length, dtype=th.long)
            dec_nodes = th.arange(tgt_length, dtype=th.long) + src_length

            # enc -> enc
            us = enc_nodes.unsqueeze(-1).repeat(1, src_length).view(-1)
            vs = enc_nodes.repeat(src_length)
            g_pool[i][j].add_edges(us, vs)
            num_edges['ee'][i][j] = len(us)
            # enc -> dec
            us = enc_nodes.unsqueeze(-1).repeat(1, tgt_length).view(-1)
            vs = dec_nodes.repeat(src_length)
            g_pool[i][j].add_edges(us, vs)
            num_edges['ed'][i][j] = len(us)
            # dec -> dec
            indices = th.triu(th.ones(tgt_length, tgt_length)) == 1
            us = dec_nodes.unsqueeze(-1).repeat(1, tgt_length)[indices]
            vs = dec_nodes.unsqueeze(0).repeat(tgt_length, 1)[indices]
            g_pool[i][j].add_edges(us, vs)
            num_edges['dd'][i][j] = len(us)

        print('successfully created graph pool, time: {0:0.3f}s'.format(time.time() - tic))
        self.g_pool = g_pool
        self.num_edges = num_edges

    def beam(self, src_buf, start_sym, max_len, k, device='cpu'):
        '''
        Return a batched graph for beam search during inference of Transformer.
        args:
            src_buf: a list of input sequence
            start_sym: the index of start-of-sequence symbol
            max_len: maximum length for decoding
            k: beam size
            device: 'cpu' or 'cuda:*' 
        '''
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [max_len] * len(src_buf)
        num_edges = {'ee': [], 'ed': [], 'dd': []}
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            for _ in range(k):
                g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))

        g = dgl.batch(g_list)
        src, tgt = [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        e2e_eids, e2d_eids, d2d_eids = [], [], []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for src_sample, n, n_ee, n_ed, n_dd in zip(src_buf, src_lens, num_edges['ee'], num_edges['ed'], num_edges['dd']):
            for _ in range(k):
                src.append(th.tensor(src_sample, dtype=th.long, device=device))
                src_pos.append(th.arange(n, dtype=th.long, device=device))
                enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
                n_nodes += n
                e2e_eids.append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))
                n_edges += n_ee
                tgt_seq = th.zeros(max_len, dtype=th.long, device=device)
                tgt_seq[0] = start_sym
                tgt.append(tgt_seq)
                tgt_pos.append(th.arange(max_len, dtype=th.long, device=device))

                dec_ids.append(th.arange(n_nodes, n_nodes + max_len, dtype=th.long, device=device))
                n_nodes += max_len
                e2d_eids.append(th.arange(n_edges, n_edges + n_ed, dtype=th.long, device=device))
                n_edges += n_ed
                d2d_eids.append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
                n_edges += n_dd

        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)
        g = g.to(device).long()

        return Graph(g=g,
                     src=(th.cat(src), th.cat(src_pos)),
                     tgt=(th.cat(tgt), th.cat(tgt_pos)),
                     tgt_y=None,
                     nids = {'enc': th.cat(enc_ids), 'dec': th.cat(dec_ids)},
                     eids = {'ee': th.cat(e2e_eids), 'ed': th.cat(e2d_eids), 'dd': th.cat(d2d_eids)},
                     nid_arr = {'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     n_tokens=n_tokens)

    def __call__(self, src_buf, tgt_buf, device='cpu'):
        '''
        Return a batched graph for the training phase of Transformer.
        args:
            src_buf: a set of input sequence arrays.
            tgt_buf: a set of output sequence arrays.
            device: 'cpu' or 'cuda:*'
        '''
        g_list = []
        src_lens = [len(_) for _ in src_buf]
        tgt_lens = [len(_) - 1 for _ in tgt_buf]
        num_edges = {'ee': [], 'ed': [], 'dd': []}
        for src_len, tgt_len in zip(src_lens, tgt_lens):
            i, j = src_len - 1, tgt_len - 1
            g_list.append(self.g_pool[i][j])
            for key in ['ee', 'ed', 'dd']:
                num_edges[key].append(int(self.num_edges[key][i][j]))

        g = dgl.batch(g_list)
        src, tgt, tgt_y = [], [], []
        src_pos, tgt_pos = [], []
        enc_ids, dec_ids = [], []
        e2e_eids, d2d_eids, e2d_eids = [], [], []
        n_nodes, n_edges, n_tokens = 0, 0, 0
        for src_sample, tgt_sample, n, m, n_ee, n_ed, n_dd in zip(src_buf, tgt_buf, src_lens, tgt_lens, num_edges['ee'], num_edges['ed'], num_edges['dd']):
            src.append(th.tensor(src_sample, dtype=th.long, device=device))
            tgt.append(th.tensor(tgt_sample[:-1], dtype=th.long, device=device))
            tgt_y.append(th.tensor(tgt_sample[1:], dtype=th.long, device=device))
            src_pos.append(th.arange(n, dtype=th.long, device=device))
            tgt_pos.append(th.arange(m, dtype=th.long, device=device))
            enc_ids.append(th.arange(n_nodes, n_nodes + n, dtype=th.long, device=device))
            n_nodes += n
            dec_ids.append(th.arange(n_nodes, n_nodes + m, dtype=th.long, device=device))
            n_nodes += m
            e2e_eids.append(th.arange(n_edges, n_edges + n_ee, dtype=th.long, device=device))
            n_edges += n_ee
            e2d_eids.append(th.arange(n_edges, n_edges + n_ed, dtype=th.long, device=device))
            n_edges += n_ed
            d2d_eids.append(th.arange(n_edges, n_edges + n_dd, dtype=th.long, device=device))
            n_edges += n_dd
            n_tokens += m


        g.set_n_initializer(dgl.init.zero_initializer)
        g.set_e_initializer(dgl.init.zero_initializer)
        g = g.to(device).long()

        return Graph(g=g,
                     src=(th.cat(src), th.cat(src_pos)),
                     tgt=(th.cat(tgt), th.cat(tgt_pos)),
                     tgt_y=th.cat(tgt_y),
                     nids = {'enc': th.cat(enc_ids), 'dec': th.cat(dec_ids)},
                     eids = {'ee': th.cat(e2e_eids), 'ed': th.cat(e2d_eids), 'dd': th.cat(d2d_eids)},
                     nid_arr = {'enc': enc_ids, 'dec': dec_ids},
                     n_nodes=n_nodes,
                     n_edges=n_edges,
                     n_tokens=n_tokens)
