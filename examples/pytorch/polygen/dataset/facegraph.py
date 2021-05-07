import dgl
import torch as th
import numpy as np
import itertools
import time
from collections import *

class FaceGraphPool:
    "Create a graph pool in advance to accelerate graph building phase in Transformer."
    def __init__(self, n=100, m=800):
        '''
        args:
            n: maximum length of input sequence.
            m: maximum length of output sequence.
        '''
        print('start creating graph pool...')
        tic = time.time()
        self.n, self.m = n, m
        g_pool = [[None for _ in range(m)] for _ in range(n)]
        num_edges = {
            'ee': np.zeros((n, m)).astype(int),
            'ed': np.zeros((n, m)).astype(int),
            'dd': np.zeros((n, m)).astype(int)
        }
        print('successfully created graph pool, time: {0:0.3f}s'.format(time.time() - tic))
        self.g_pool = g_pool
        self.num_edges = num_edges


    def get_graph_for_size(self, i, j):
        '''
        Lazy evaluation of the graph[i][j]
        args:
            i: encoder size
            j: decoder size, note it should be exact len seq + 1, the one is for eos token
        '''
        if self.g_pool[i][j]:
            return self.g_pool[i][j]

        src_length = i
        # Decoder need +1 for the start token
        tgt_length = j
        
        self.g_pool[i][j] = dgl.DGLGraph()
        self.g_pool[i][j].add_nodes(src_length + tgt_length)
        enc_nodes = th.arange(src_length, dtype=th.long)
        dec_nodes = th.arange(tgt_length, dtype=th.long) + src_length

        # enc -> enc
        us = enc_nodes.unsqueeze(-1).repeat(1, src_length).view(-1)
        vs = enc_nodes.repeat(src_length)
        self.g_pool[i][j].add_edges(us, vs)
        self.num_edges['ee'][i][j] = len(us)
        # enc -> dec
        us = enc_nodes.unsqueeze(-1).repeat(1, tgt_length).view(-1)
        vs = dec_nodes.repeat(src_length)
        self.g_pool[i][j].add_edges(us, vs)
        self.num_edges['ed'][i][j] = len(us)
        # dec -> dec
        indices = th.triu(th.ones(tgt_length, tgt_length)) == 1
        us = dec_nodes.unsqueeze(-1).repeat(1, tgt_length)[indices]
        vs = dec_nodes.unsqueeze(0).repeat(tgt_length, 1)[indices]
        self.g_pool[i][j].add_edges(us, vs)
        self.num_edges['dd'][i][j] = len(us)
        return self.g_pool[i][j]