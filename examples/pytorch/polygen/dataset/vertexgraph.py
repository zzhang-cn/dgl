import dgl
import torch as th
import numpy as np
import itertools
import time
from collections import *

class VertexNetGraphPool:
    "Create a graph pool in advance to accelerate graph building phase in Transformer."
    def __init__(self, n=400):
        '''
        args:
            n: maximum number of vertexes
        '''
        print('start creating graph pool...')
        tic = time.time()
        self.n = n
        g_pool = [None for _ in range(n)]
        num_edges = {
            'dd': np.zeros(n).astype(int)
        }
        print('successfully created graph pool, time: {0:0.3f}s'.format(time.time() - tic))
        self.g_pool = g_pool
        self.num_edges = num_edges

    def get_graph_for_size(self, i):
        '''
        Lazy evaluation of the graph[i]
        args:
            i: number of nodes, should be exact len + 1, the one is for eos token
        '''
        if self.g_pool[i]:
            return self.g_pool[i]

        n_vertex = i

        self.g_pool[i] = dgl.DGLGraph()
        self.g_pool[i].add_nodes(n_vertex)
        dec_nodes = th.arange(n_vertex, dtype=th.long)

        # dec -> dec
        indices = th.triu(th.ones(n_vertex, n_vertex)) == 1
        us = dec_nodes.unsqueeze(-1).repeat(1, n_vertex)[indices]
        vs = dec_nodes.unsqueeze(0).repeat(n_vertex, 1)[indices]
        self.g_pool[i].add_edges(us, vs)
        self.num_edges['dd'][i] = len(us)
        return self.g_pool[i]